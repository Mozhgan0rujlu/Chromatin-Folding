
"""


@author: mozhganoroujlu
"""


import os
import csv
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compute compartment energy from paired contact and quantile score files.")
    parser.add_argument(
        "contacts_folder",
        default="filtered_tads_contacts/", #prefiltered intra-TADs contacts
        nargs="?",
        help="Folder with *_filtered.txt contact files"
    )
    parser.add_argument(
        "scores_folder",
        default="compartment_scores/", #precomputed compartment scores
        nargs="?",
        help="Folder with barcode.txt score files (2 columns: bin_id score)"
    )
    parser.add_argument(
        "output_csv",
        default="new_energy.csv",
        nargs="?",
        help="Output CSV path"
    )
    args = parser.parse_args()

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    results = []
    missing_scores = []
    processed = 0

    for filename in os.listdir(args.contacts_folder):
        if not filename.endswith("_filtered.txt"):
            continue

        barcode = filename.replace("_filtered.txt", "")
        contact_path = os.path.join(args.contacts_folder, filename)
        score_path = os.path.join(args.scores_folder, barcode + ".txt")

        if not os.path.exists(score_path):
            print(f"Warning: Missing score file: {score_path}")
            missing_scores.append(barcode)
            continue

        # Read scores: bin_id -> score (2-column format)
        scores = {}
        try:
            with open(score_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        print(f"    Warning: Skipping malformed line {line_num} in {score_path}: {line}")
                        continue
                    bin_id = int(parts[0])
                    score = float(parts[1])
                    scores[bin_id] = score
        except Exception as e:
            print(f"Error reading score file {score_path}: {e}")
            missing_scores.append(barcode)
            continue

        # Now process contacts and compute energy
        sum_prod = 0.0
        contact_count = 0
        try:
            with open(contact_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 3:
                        continue  # skip header or malformed lines
                    i, j, c = int(parts[0]), int(parts[1]), float(parts[2])

                    if i in scores and j in scores:
                        sum_prod += scores[i] * scores[j] * c
                        contact_count += 1
        except Exception as e:
            print(f"Error reading contact file {contact_path}: {e}")
            continue

        energy = -sum_prod  # This is the "Ising energy"
        results.append((barcode, energy))
        processed += 1
        print(f"Processed {barcode}: {contact_count} contacts, Energy = {energy:.6f}")

    # Write results
    try:
        with open(args.output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['barcode', 'Energy'])
            for barcode, energy in sorted(results):  # optional: sort by barcode
                writer.writerow([barcode, energy])
        print(f"\nSuccess! Output written to: {args.output_csv}")
        print(f"   {processed} cells processed successfully")
    except Exception as e:
        print(f"Error writing CSV: {e}")

    if missing_scores:
        print(f"\nWarning: {len(missing_scores)} barcodes had missing score files:")
        print("   " + ", ".join(sorted(missing_scores)))

if __name__ == "__main__":
    main()
