
import qrcode
import os
import pandas as pd

# Function to generate QR code
def generate_qr_code(url, output_path):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_path)

# Function to generate QR code
def generate_qr_code(url, output_path):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_path)

# Function to create directories if they don't exist
def create_directories(good_dir, bad_dir):
    if not os.path.exists(good_dir):
        os.makedirs(good_dir)
    if not os.path.exists(bad_dir):
        os.makedirs(bad_dir)

# Read CSV file
csv_file = './phishing_site_urls.csv' # Change this to your CSV file name
df = pd.read_csv(csv_file)
# Output directories
good_dir = './QR_codes/good_qr_codes'
bad_dir = './QR_codes/bad_qr_codes'
create_directories(good_dir, bad_dir)

# Generate QR codes
for index, row in df.iterrows():
    url = row['URL']
    label = row['Label'] # Assuming column name is 'Label'
    output_path = os.path.join(good_dir if label == 'good' else bad_dir, f'{index}.png')
    generate_qr_code(url, output_path)
print("QR codes generated successfully.")