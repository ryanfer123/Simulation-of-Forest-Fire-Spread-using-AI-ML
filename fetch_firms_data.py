import urllib.request
import os

def download_firms_data():
    url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_7d.csv"
    os.makedirs("data", exist_ok=True)
    output_path = "data/suomi_viirs_7d.csv"
    
    print(f"Downloading NASA FIRMS data from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Successfully downloaded to {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    except Exception as e:
        print(f"Failed to download data: {e}")

if __name__ == "__main__":
    download_firms_data()
