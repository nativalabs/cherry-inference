from google.cloud import secretmanager
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "kedar@nativatraders.com"
SMTP_PASSWORD_SECRET_NAME = "smtp-kedar-password-secret"

email_recipients = ["gabriel@nativatraders.com","kedar@nativatraders.com", "juana@nativatraders.com"]
recipient_list = ", ".join(email_recipients)

def access_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/913436042986/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": secret_name})
    return response.payload.data.decode("UTF-8")

smtp_password = access_secret(SMTP_PASSWORD_SECRET_NAME)

def add_to_mail(msg,content, content_type, name):
    if content_type == 'json':
        json_attachment = MIMEText(content, 'plain')
        json_attachment.add_header('Content-Disposition', f'attachment; filename={name}')
        msg.attach(json_attachment)
    elif content_type == 'image':
        mail_image = MIMEImage(content, name=name)
        msg.attach(mail_image)
    elif content_type == 'text':
        msg.attach(MIMEText(content, 'plain'))