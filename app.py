from flask import Flask, render_template, request, jsonify, redirect, Response
import requests
import json
import os
from datetime import datetime
import random
import hashlib
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import uuid
import csv
import io
import re

# Import psycopg2 for PostgreSQL
import psycopg2
from psycopg2 import sql
from urllib.parse import urlparse

app = Flask(__name__)

# --- Configuration for Render Deployment ---
PORT = int(os.environ.get("PORT", 5000))
if os.environ.get("RENDER_EXTERNAL_HOSTNAME"):
    BASE_URL = f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME')}"
else:
    BASE_URL = f"http://localhost:{PORT}"

print(f"Application will use BASE_URL: {BASE_URL}")

# --- Database Configuration (PostgreSQL) ---
# Render provides DATABASE_URL for PostgreSQL services
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    if not DATABASE_URL:
        # For local development, if you want to use a local PostgreSQL without Render's DATABASE_URL:
        # You would replace this with your local PostgreSQL connection details
        print("DATABASE_URL environment variable not set. Please set it for production deployment.")
        # As a fallback for local testing without setting DATABASE_URL, you could provide static credentials
        # Or, raise an error to force setting the variable.
        raise ValueError("DATABASE_URL environment variable is not set. Cannot connect to PostgreSQL.")

    # Parse the DATABASE_URL provided by Render (e.g., postgresql://user:password@host:port/database)
    result = urlparse(DATABASE_URL)
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    port = result.port

    conn = psycopg2.connect(
        host = hostname,
        port = port,
        database = database,
        user = username,
        password = password,
        sslmode='require' # Add this for Render PostgreSQL connections
    )
    return conn

# Database initialization (PostgreSQL specific SQL)
def init_db():
    """Initialize PostgreSQL database for A/B testing tracking"""
    conn = None # Initialize conn to None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Campaigns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS campaigns (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                company_name TEXT NOT NULL,
                product_name TEXT NOT NULL,
                offer_details TEXT NOT NULL,
                campaign_type TEXT NOT NULL,
                target_audience TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'draft',
                total_recipients INTEGER DEFAULT 0
            )
        ''')

        # Email variations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_variations (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                variation_name TEXT NOT NULL,
                subject_line TEXT NOT NULL,
                email_body TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (campaign_id) REFERENCES campaigns (id) ON DELETE CASCADE
            )
        ''')

        # Recipients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipients (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                email_address TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                variation_assigned TEXT NOT NULL,
                sent_at TIMESTAMP,
                opened_at TIMESTAMP,
                clicked_at TIMESTAMP,
                converted_at TIMESTAMP,
                status TEXT DEFAULT 'pending',
                tracking_id TEXT UNIQUE,
                FOREIGN KEY (campaign_id) REFERENCES campaigns (id) ON DELETE CASCADE
            )
        ''')

        # A/B test results table (Note: PostgreSQL uses SERIAL for auto-incrementing integers)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_results (
                id SERIAL PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                variation_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (campaign_id) REFERENCES campaigns (id) ON DELETE CASCADE
            )
        ''')

        conn.commit()
        cursor.close()
        print("PostgreSQL database tables checked/created successfully!")
    except Exception as e:
        print(f"Error initializing PostgreSQL database: {e}")
        # Depending on criticality, you might want to exit or raise the exception.
        # For a web app, a failed DB init usually means the app can't function.
        raise # Re-raise the exception so Render logs it as a fatal error
    finally:
        if conn:
            conn.close()


# --- Call init_db() immediately after app creation ---
# This ensures tables are created when the app starts, regardless of how it's run (gunicorn or direct python)
try:
    init_db()
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize database: {e}")
    # In a real production app, you might want a more graceful shutdown or alert system
    # For now, we let the exception propagate so Render knows the service failed to start.


# Gmail API configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/gmail.readonly']

# Hugging Face API configuration
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{LLAMA_MODEL}"
# Fetch token from environment, provide a dummy default for local testing if not set
HF_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN', 'your_huggingface_api_token_here_if_testing_locally_without_env_var')

# Ensure credentials.json and token.json are present from environment variables
# This block should be placed at the top level of your script, after 'app = Flask(__name__)'
# These files are transiently created on Render from env vars for the Gmail API to use.
if os.environ.get('GOOGLE_CREDENTIALS_JSON_B64'):
    try:
        decoded_credentials = base64.b64decode(os.environ['GOOGLE_CREDENTIALS_JSON_B64']).decode('utf-8')
        with open('credentials.json', 'w') as f:
            f.write(decoded_credentials)
        print("credentials.json created from environment variable.")
    except Exception as e:
        print(f"Error decoding GOOGLE_CREDENTIALS_JSON_B64: {e}")

if os.environ.get('GOOGLE_TOKEN_JSON_B64'):
    try:
        decoded_token = base64.b64decode(os.environ['GOOGLE_TOKEN_JSON_B64']).decode('utf-8')
        with open('token.json', 'w') as f:
            f.write(decoded_token)
        print("token.json created from environment variable.")
    except Exception as e:
        print(f"Error decoding GOOGLE_TOKEN_JSON_B64: {e}")

# Gmail API functions
def authenticate_gmail():
    """Authenticate and return Gmail service object"""
    creds = None

    # Load existing credentials
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If no valid credentials, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # You need to download credentials.json from Google Cloud Console
            if os.path.exists('credentials.json'):
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            else:
                raise Exception("credentials.json file not found. Download it from Google Cloud Console or set GOOGLE_CREDENTIALS_JSON_B64.")

        # Save credentials for next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def create_email_message(to_email, subject, body, tracking_id):
    """Create email message with tracking pixel"""
    message = MIMEMultipart('alternative')
    message['to'] = to_email
    message['subject'] = subject

    # Add tracking pixel to HTML version
    tracking_pixel = f'<img src="{BASE_URL}/pixel/{tracking_id}" width="1" height="1" style="display:none;">'

    # Convert plain text body to HTML and add tracking
    html_body = body.replace('\n', '<br>') + tracking_pixel

    # Add click tracking to links
    html_body = add_click_tracking(html_body, tracking_id)

    # Create both plain text and HTML versions
    text_part = MIMEText(body, 'plain')
    html_part = MIMEText(html_body, 'html')

    message.attach(text_part)
    message.attach(html_part)

    return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

def add_click_tracking(html_body, tracking_id):
    """Add click tracking to links in email body"""
    # Find all links and replace with tracking links
    def replace_link(match):
        original_url = match.group(1)
        tracking_url = f"{BASE_URL}/click/{tracking_id}?url={original_url}"
        return f'href="{tracking_url}"'

    # Replace href attributes
    html_body = re.sub(r'href="([^"]*)"', replace_link, html_body)

    return html_body

def send_email_via_gmail(service, email_message):
    """Send email using Gmail API"""
    try:
        message = service.users().messages().send(userId="me", body=email_message).execute()
        return {'success': True, 'message_id': message['id']}
    except HttpError as error:
        return {'success': False, 'error': str(error)}

# A/B Testing functions
def assign_variation(recipient_email, variations):
    """Assign recipient to a variation using consistent hashing"""
    # Use email hash to ensure consistent assignment
    email_hash = hashlib.md5(recipient_email.encode()).hexdigest()
    hash_int = int(email_hash[:8], 16)
    variation_index = hash_int % len(variations)
    return variations[variation_index]['variation_name']

def calculate_ab_metrics(campaign_id):
    """Calculate A/B testing metrics for a campaign"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all variations for this campaign
    cursor.execute(sql.SQL('''
        SELECT DISTINCT variation_assigned FROM recipients
        WHERE campaign_id = %s
    '''), [campaign_id])
    variations = [row[0] for row in cursor.fetchall()]

    metrics = {}

    for variation in variations:
        # Calculate metrics for each variation
        cursor.execute(sql.SQL('''
            SELECT
                COUNT(*) as total_sent,
                COUNT(CASE WHEN opened_at IS NOT NULL THEN 1 END) as opened,
                COUNT(CASE WHEN converted_at IS NOT NULL THEN 1 END) as converted,
                COUNT(CASE WHEN opened_at IS NOT NULL AND (SELECT COUNT(*) FROM recipients WHERE campaign_id = %s AND variation_assigned = %s AND email_address = r.email_address AND opened_at IS NOT NULL) > 1 THEN 1 END) as re_opened
            FROM recipients r
            WHERE campaign_id = %s AND variation_assigned = %s AND status = 'sent'
        '''), [campaign_id, variation, campaign_id, variation]) #

        result = cursor.fetchone()
        total_sent, opened, converted, re_opened = result #

        metrics[variation] = {
            'total_sent': total_sent, #
            'opened': opened, #
            'converted': converted, #
            'open_rate': (opened / total_sent * 100) if total_sent > 0 else 0, #
            'conversion_rate': (converted / total_sent * 100) if total_sent > 0 else 0, #
            're_open_rate': (re_opened / opened * 100) if opened > 0 else 0 # New re-open rate
        }

    cursor.close()
    conn.close()
    return metrics

# Original email generation functions (keeping existing code)
def query_huggingface(payload):
    """Query the Hugging Face API using Llama 3 8B"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"error": "Model is loading. Please wait and try again."}
        elif response.status_code == 404:
            return {"error": "Model not accessible. Check API token permissions."}
        elif response.status_code == 429:
            return {"error": "Rate limit exceeded. Please wait before trying again."}
        else:
            return {"error": f"API request failed with status {response.status_code}"}

    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

def generate_email_variations(company_name, product_name, offer_details, campaign_type, target_audience=""):
    """Generate email variations using AI"""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert email marketing copywriter. Create two completely different marketing email variations for A/B testing. Each should have a unique approach and tone while maintaining high conversion potential.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Create TWO different marketing email variations for A/B testing:

Company: {company_name}
Product/Service: {product_name}
Campaign Focus: {offer_details}
Type: {campaign_type}
Audience: {target_audience if target_audience else "General customers"}

Requirements:
- Subject line: Under 50 characters, A/B test friendly
- Email body: Professional, persuasive, conversion-focused
- Minimal emoji use (2-3 maximum per email)
- Different psychological triggers for each variation
- Clear call-to-action with trackable links
- Optimized for mobile reading

Format response as:

VARIATION A:
SUBJECT: [subject line]
BODY: [email content]

VARIATION B:
SUBJECT: [subject line]
BODY: [email content]

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
            "stop": ["<|eot_id|>"]
        }
    }

    result = query_huggingface(payload)

    if 'error' in result:
        print(f"Hugging Face API Error: {result['error']}. Generating fallback variations.")
        return create_fallback_variations(company_name, product_name, offer_details, campaign_type)

    return result

def create_fallback_variations(company_name, product_name, offer_details, campaign_type):
    """Create fallback variations optimized for A/B testing"""

    variation_a = {
        'subject': f'🚀 {product_name} - Limited Time',
        'body': f'''Hi there,

Big news! We've just launched {product_name} and it's already creating a buzz.

{offer_details}

Here's what makes this special:
✓ Designed specifically for people like you
✓ Proven results from our beta testing
✓ Limited-time exclusive access

Ready to be among the first to experience this?

[Claim Your Spot Now]

Best,
{company_name} Team

P.S. This offer expires soon - don't miss out!'''
    }

    variation_b = {
        'subject': f'You\'re invited: {product_name}',
        'body': f'''Hello!

We have something exciting to share with you.

After months of development, {product_name} is finally here. The early feedback has been incredible, and we think you'll love what we've created.

{offer_details}

What our customers are saying:
"This exceeded all my expectations" - Sarah M.
"Finally, a solution that actually works" - David L.

Want to see what all the excitement is about?

[Discover More]

Warmly,
The {company_name} Team

P.S. Join hundreds of satisfied customers who've already made the switch. 🌟'''
    }

    return [{"generated_text": f"VARIATION A:\nSUBJECT: {variation_a['subject']}\nBODY: {variation_a['body']}\n\nVARIATION B:\nSUBJECT: {variation_b['subject']}\nBODY: {variation_b['body']}"}]

# API Routes
@app.route('/')
def index():
    """Redirects to the A/B Dashboard"""
    return redirect('/ab-dashboard')

@app.route('/ab-dashboard')
def ab_dashboard():
    """A/B testing dashboard"""
    return render_template('ab_dashboard.html', base_url=BASE_URL)

@app.route('/create-campaign', methods=['POST'])
def create_campaign():
    """Create a new A/B testing campaign"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['company_name', 'product_name', 'offer_details', 'campaign_type']
        if not all(field in data and data[field].strip() for field in required_fields):
            return jsonify({'success': False, 'error': 'Missing required fields'})

        # Generate email variations
        result = generate_email_variations(
            data['company_name'], data['product_name'],
            data['offer_details'], data['campaign_type'],
            data.get('target_audience', '')
        )

        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']})

        # Parse variations
        variations = parse_email_variations(result[0]['generated_text'])

        # Create campaign in database
        conn = get_db_connection()
        cursor = conn.cursor()

        campaign_id = str(uuid.uuid4())
        cursor.execute(sql.SQL('''
            INSERT INTO campaigns (id, name, company_name, product_name, offer_details, campaign_type, target_audience)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        '''), (
            campaign_id,
            f"{data['company_name']} - {data['campaign_type'].title()}",
            data['company_name'],
            data['product_name'],
            data['offer_details'],
            data['campaign_type'],
            data.get('target_audience', '')
        ))

        # Save variations
        for i, variation in enumerate(variations):
            variation_id = str(uuid.uuid4())
            cursor.execute(sql.SQL('''
                INSERT INTO email_variations (id, campaign_id, variation_name, subject_line, email_body)
                VALUES (%s, %s, %s, %s, %s)
            '''), (
                variation_id, campaign_id, f"Variation_{chr(65+i)}",
                variation['subject'], variation['body']
            ))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'campaign_id': campaign_id,
            'variations': variations
        })

    except Exception as e:
        print(f"Error in create_campaign: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload-recipients', methods=['POST'])
def upload_recipients():
    """Upload recipient list for A/B testing"""
    try:
        campaign_id = request.form.get('campaign_id')
        if not campaign_id:
            return jsonify({'success': False, 'error': 'Campaign ID required'})

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        # Read CSV file
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.DictReader(stream)

        # Get campaign variations
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(sql.SQL('SELECT variation_name FROM email_variations WHERE campaign_id = %s'), [campaign_id])
        variations = [{'variation_name': row[0]} for row in cursor.fetchall()]

        recipients_added = 0

        for row in csv_input:
            email = row.get('email', '').strip()
            if not email:
                continue

            # Assign variation
            assigned_variation = assign_variation(email, variations)
            tracking_id = str(uuid.uuid4())

            cursor.execute(sql.SQL('''
                INSERT INTO recipients (id, campaign_id, email_address, first_name, last_name, variation_assigned, tracking_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            '''), (
                str(uuid.uuid4()), campaign_id, email,
                row.get('first_name', ''), row.get('last_name', ''),
                assigned_variation, tracking_id
            ))
            recipients_added += 1

        # Update campaign total recipients
        cursor.execute(sql.SQL('UPDATE campaigns SET total_recipients = %s WHERE id = %s'), (recipients_added, campaign_id))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'recipients_added': recipients_added,
            'message': f'Successfully uploaded {recipients_added} recipients'
        })

    except Exception as e:
        print(f"Error in upload_recipients: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/send-campaign', methods=['POST'])
def send_campaign():
    """Send A/B testing campaign"""
    try:
        data = request.get_json()
        campaign_id = data.get('campaign_id')

        if not campaign_id:
            return jsonify({'success': False, 'error': 'Campaign ID required'})

        # Authenticate Gmail
        try:
            gmail_service = authenticate_gmail()
        except Exception as e:
            return jsonify({'success': False, 'error': f'Gmail authentication failed: {str(e)}'})

        # Get campaign and variations
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get email variations
        cursor.execute(sql.SQL('''
            SELECT variation_name, subject_line, email_body
            FROM email_variations
            WHERE campaign_id = %s
        '''), [campaign_id])
        variations = {row[0]: {'subject': row[1], 'body': row[2]} for row in cursor.fetchall()}

        # Get recipients
        cursor.execute(sql.SQL('''
            SELECT id, email_address, first_name, variation_assigned, tracking_id
            FROM recipients
            WHERE campaign_id = %s AND status = 'pending'
        '''), [campaign_id])
        recipients = cursor.fetchall()

        sent_count = 0
        errors = []

        print(f"--- Starting to send campaign {campaign_id} to {len(recipients)} recipients ---")

        for recipient_id, email, first_name, variation, tracking_id in recipients:
            print(f"\nProcessing recipient: {email} for variation: {variation}")
            try:
                # Get variation content
                variation_content = variations[variation]

                # Personalize content
                subject = variation_content['subject']
                body = variation_content['body']
                if first_name:
                    body = body.replace('Hi there', f'Hi {first_name}')
                    body = body.replace('Hello!', f'Hello {first_name}!')

                # Create and send email
                print(f"  > Creating email message for {email}...")
                email_message = create_email_message(email, subject, body, tracking_id)

                print(f"  > Attempting to send via Gmail API...")
                result = send_email_via_gmail(gmail_service, email_message)

                if result['success']:
                    print(f"  > SUCCESS: Email sent. Updating status to 'sent'.")
                    # Update recipient status
                    cursor.execute(sql.SQL('''
                        UPDATE recipients
                        SET status = 'sent', sent_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    '''), [recipient_id])
                    sent_count += 1
                else:
                    print(f"  > FAILED: Gmail API returned an error: {result['error']}")
                    errors.append(f'{email}: {result["error"]}')
                    cursor.execute(sql.SQL('''
                        UPDATE recipients
                        SET status = 'failed'
                        WHERE id = %s
                    '''), [recipient_id])

            except Exception as e:
                print(f"  > FAILED: An exception occurred: {str(e)}")
                errors.append(f'{email}: {str(e)}')

        # Commit all the database changes at the end of the loop
        conn.commit()

        print(f"--- Campaign sending finished. Committing changes to database. ---")

        # Update campaign status
        cursor.execute(sql.SQL('UPDATE campaigns SET status = %s WHERE id = %s'), ('sent', campaign_id))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'sent_count': sent_count,
            'total_recipients': len(recipients),
            'errors': errors[:10]  # Limit error list
        })

    except Exception as e:
        print(f"Error in send_campaign: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/campaign-results/<campaign_id>')
def campaign_results(campaign_id):
    """Get A/B testing results for a campaign"""
    conn = None # Initialize conn to None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(sql.SQL('SELECT name, status, total_recipients FROM campaigns WHERE id = %s'), [campaign_id])
        campaign = cursor.fetchone()

        cursor.close()
        # conn.close() # Close conn in finally block

        if not campaign:
            return jsonify({'success': False, 'error': 'Campaign not found'})

        metrics = calculate_ab_metrics(campaign_id) # This function gets its own connection

        return jsonify({
            'success': True,
            'campaign': {
                'name': campaign[0],
                'status': campaign[1],
                'total_recipients': campaign[2]
            },
            'metrics': metrics
        })

    except Exception as e:
        print(f"Error in campaign_results: {e}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if conn:
            conn.close()

@app.route('/campaigns')
def list_campaigns():
    """List all campaigns"""
    conn = None # Initialize conn to None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(sql.SQL('SELECT id, name, status, total_recipients, created_at FROM campaigns ORDER BY created_at DESC'))
        campaigns = [
            {
                'id': row[0],
                'name': row[1],
                'status': row[2],
                'total_recipients': row[3],
                'created_at': row[4]
            }
            for row in cursor.fetchall()
        ]

        cursor.close()
        # conn.close() # Close conn in finally block

        return jsonify({'success': True, 'campaigns': campaigns})

    except Exception as e:
        print(f"Error in list_campaigns: {e}")
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if conn:
            conn.close()


# Tracking routes
@app.route('/pixel/<tracking_id>')
def tracking_pixel(tracking_id):
    """Track email opens"""
    conn = None # Initialize conn to None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(sql.SQL('''
            UPDATE recipients
            SET opened_at = CURRENT_TIMESTAMP
            WHERE tracking_id = %s AND opened_at IS NULL
        '''), [tracking_id])

        conn.commit()
        cursor.close()
        # conn.close() # Close conn in finally block

        # Return 1x1 transparent pixel
        pixel = base64.b64decode('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7')
        return Response(pixel, mimetype='image/gif')

    except Exception as e:
        print(f"Error tracking pixel for {tracking_id}: {e}")
        # Return pixel even if tracking fails, to not break email client display
        pixel = base64.b64decode('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7')
        return Response(pixel, mimetype='image/gif')
    finally:
        if conn:
            conn.close()

@app.route('/click/<tracking_id>')
def track_click(tracking_id):
    """Track email clicks and redirect"""
    conn = None # Initialize conn to None
    try:
        original_url = request.args.get('url', BASE_URL) # Fallback to BASE_URL

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(sql.SQL('''
            UPDATE recipients
            SET clicked_at = CURRENT_TIMESTAMP
            WHERE tracking_id = %s AND clicked_at IS NULL
        '''), [tracking_id])

        conn.commit()
        cursor.close()
        # conn.close() # Close conn in finally block

        return redirect(original_url)

    except Exception as e:
        print(f"Error tracking click for {tracking_id}: {e}")
        return redirect(BASE_URL) # Redirect to BASE_URL on error
    finally:
        if conn:
            conn.close()

def parse_email_variations(generated_text):
    """Parse generated text into variation objects"""
    variations = []

    # Split by VARIATION markers
    parts = generated_text.split('VARIATION')

    for i, part in enumerate(parts[1:], 1):
        if i > 2:
            break # Only take first two variations

        lines = part.strip().split('\n')
        subject = ""
        body_lines = []
        body_started = False

        for line in lines:
            line = line.strip()
            if line.upper().startswith('SUBJECT:'):
                subject = line[8:].strip()
            elif line.upper().startswith('BODY:'):
                body_started = True
            elif body_started: # Any non-empty line after BODY: is part of the body
                body_lines.append(line)

        body = '\n'.join(body_lines).strip() # .strip() removes leading/trailing whitespace

        if subject and body:
            variations.append({
                'subject': subject,
                'body': body
            })

    # Fallback if parsing fails or less than 2 variations are generated
    if len(variations) < 2:
        print("Warning: Less than two variations parsed. Using fallback variations.")
        variations = [
            {
                'subject': 'Exclusive Offer Inside 🎯',
                'body': 'We have something special for you...\n\n[Learn More]'
            },
            {
                'subject': 'You\'re Going to Love This',
                'body': 'This is exactly what you\'ve been waiting for...\n\n[Discover More]'
            }
        ]

    return variations

if __name__ == '__main__':
    # This block will now only run when you execute 'python final.py' directly.
    # The init_db() call for Gunicorn is moved above.
    print("🧪 A/B Testing Email Marketing App")
    print("✉️  Gmail API Integration Ready")
    print("📊 Campaign Tracking Enabled")
    print("🎯 Endpoints:")
    print(f"   - Main: {BASE_URL}")
    print(f"   - Dashboard: {BASE_URL}/ab-dashboard")
    print(f"   - Campaigns: {BASE_URL}/campaigns")

    app.run(debug=True, host='0.0.0.0', port=PORT)
