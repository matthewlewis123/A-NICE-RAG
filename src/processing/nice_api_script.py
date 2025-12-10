import json
import os
import time

import requests


def download_all_guidelines():
    """Extracts all guideline numbers from the API, then downloads all XML files."""

    # --- API Configuration ---
    api_key = "YOUR_API_KEY_HERE"
    index_url = "https://api.nice.org.uk/services/guidance/index/current.json"
    base_guidance_url = "https://api.nice.org.uk/services/guidance/structured-documents/"

    headers = {"API-Key": api_key, "Accept": "application/json"}

    xml_headers = {"API-Key": api_key, "Accept": "application/xml"}

    print("Fetching guideline index from current.json")
    try:
        response = requests.get(index_url, headers=headers)
        response.raise_for_status()
        guideline_data = response.json()

        guideline_list = guideline_data.get("IndexItems", [])
        if not guideline_list:
            print("Could not find 'IndexItems' in the API response.")
            return

        print(f"Found {len(guideline_list)} total items in the index.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching guideline index: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return

    guideline_numbers = []
    for g in guideline_list:
        guidance_number = g.get("GuidanceNumber", "")
        if guidance_number:
            guideline_numbers.append(
                {
                    "number": guidance_number,
                    "title": g.get("Title", "No Title"),
                    "programme": g.get("GuidanceProgramme", "Unknown"),
                }
            )

    print(f"Found {len(guideline_numbers)} guidelines with valid numbers to download.")

    output_directory = "NICE_Guidelines_XML"
    os.makedirs(output_directory, exist_ok=True)

    successful_downloads = 0
    failed_downloads = 0

    for i, guideline in enumerate(guideline_numbers, 1):
        guidance_number = guideline["number"]

        print(f"\n--- Processing {i}/{len(guideline_numbers)}: {guidance_number} ---")

        guideline_url = f"{base_guidance_url}{guidance_number}"
        output_filename = f"{guidance_number}_structured_document.xml"
        output_filepath = os.path.join(output_directory, output_filename)

        print(f"Attempting to retrieve XML from: {guideline_url}")

        try:
            response = requests.get(guideline_url, headers=xml_headers)
            response.raise_for_status()

            if response.status_code == 200:
                with open(output_filepath, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"✓ Successfully saved to {output_filepath}")
                successful_downloads += 1
            elif response.status_code == 304:
                print("Resource not modified since last request.")
                successful_downloads += 1
            elif response.status_code == 401:
                print("✗ Unauthorized: API key issue")
                failed_downloads += 1
            elif response.status_code == 403:
                print("✗ Forbidden: Access denied")
                failed_downloads += 1
            elif response.status_code == 404:
                print("✗ Not Found: No structured document available")
                failed_downloads += 1
            else:
                print(f"✗ Request failed with status code: {response.status_code}")
                failed_downloads += 1

        except requests.exceptions.RequestException as e:
            print(f"✗ An error occurred: {e}")
            failed_downloads += 1
        except IOError as e:
            print(f"✗ Error saving file: {e}")
            failed_downloads += 1

        time.sleep(0.5)

    print("Summary")
    print(f"Total guidelines processed: {len(guideline_numbers)}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Files saved to: {output_directory}")


# --- Execute ---
download_all_guidelines()
