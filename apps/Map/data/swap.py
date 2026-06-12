import json
import os
import sys

def main():
    # Adjust path if needed
    file_path = "places.json"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    # Create a backup
    backup_path = file_path + ".backup"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy(file_path, backup_path)
        print(f"Backup saved to {backup_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure search_index exists
    if "search_index" not in data:
        data["search_index"] = {}
        print("Created empty search_index")

    search_index = data["search_index"]
    places = data.get("places", {})
    added = 0

    for place_id, place in places.items():
        # Get English name
        name_field = place.get("name")
        if isinstance(name_field, dict) and "en" in name_field:
            eng_name = name_field["en"].strip()
            if eng_name:
                if eng_name not in search_index:
                    search_index[eng_name] = []
                if place_id not in search_index[eng_name]:
                    search_index[eng_name].append(place_id)
                    added += 1
                    print(f"Added search index entry: '{eng_name}' -> {place_id}")

        # Also add English part of formattedAddress if you want (optional)
        # address_field = place.get("formattedAddress")
        # if isinstance(address_field, dict) and "en" in address_field:
        #     eng_addr = address_field["en"].strip()
        #     if eng_addr and eng_addr not in search_index:
        #         search_index[eng_addr] = [place_id]
        #         added += 1

        # Optional: add English displayName from details
        details = place.get("details")
        if isinstance(details, dict):
            display = details.get("displayName")
            if isinstance(display, dict) and "en" in display:
                eng_disp = display["en"].strip()
                if eng_disp and eng_disp not in search_index:
                    search_index[eng_disp] = [place_id]
                    added += 1
                    print(f"Added search index from displayName: '{eng_disp}' -> {place_id}")

    if added == 0:
        print("No new English entries added to search_index.")
    else:
        # Save the modified JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Added {added} English keys to search_index.")
        print("Now rebuild the frontend: npm run build && npm run dev")

if __name__ == "__main__":
    main()