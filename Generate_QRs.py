
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

def get_small_set(data):
    #2.5 to 1 ratio
    good_chunk = data[data['Label'] == 'good']
    bad_chunk = data[data['Label'] == 'bad']
    both = pd.concat([good_chunk.iloc[:35000], bad_chunk.iloc[:15000]], ignore_index=True)
    output = both.sample(frac=1).reset_index(drop=True)
    return output

def get_micro_set(data):
    micro_data = data[data['URL'].str.len() < 26]
    micro_data = micro_data.reset_index()
    return micro_data

# Read CSV file
csv_file = './phishing_site_urls.csv' # Change this to your CSV file name
df = pd.read_csv(csv_file)
# Output directories
good_dir = './micro_QR_codes/good_qr_codes'
bad_dir = './micro_QR_codes/bad_qr_codes'
create_directories(good_dir, bad_dir)
df = get_micro_set(df)

# Generate QR codes
for index, row in df.iterrows():
    url = row['URL']
    label = row['Label'] # Assuming column name is 'Label'
    output_path = os.path.join(good_dir if label == 'good' else bad_dir, f'{index}.png')
    generate_qr_code(url, output_path)
print("QR codes generated successfully.")