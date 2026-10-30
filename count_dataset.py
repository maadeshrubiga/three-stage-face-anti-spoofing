import os
import glob
from collections import defaultdict
from tabulate import tabulate  # You might need to install this or I can format manually

# Define extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
VID_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

def get_category_counts(root_path):
    stats = defaultdict(lambda: {'images': 0, 'videos': 0})
    
    if not os.path.exists(root_path):
        return None

    # We want to count by "Type" (immediate subfolder)
    # e.g. dataset/live/UTKFace -> Type: UTKFace
    
    # Just walking the root folder
    for root, dirs, files in os.walk(root_path):
        # Determine the "Type"
        # If root represents a subfolder of root_path, use that.
        # If root == root_path, files are "Uncategorized" there.
        
        rel_path = os.path.relpath(root, root_path)
        
        if rel_path == '.':
            type_name = "Root Files"
        else:
            # use the top-level folder name as the type
            parts = rel_path.split(os.sep)
            type_name = parts[0]
            
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                stats[type_name]['images'] += 1
            elif ext in VID_EXTS:
                stats[type_name]['videos'] += 1
                
    return stats

def main():
    base_dir = r"c:\Users\manos\Videos\Final Year\dataset"
    if not os.path.exists(base_dir):
        print(f"Error: Dataset directory not found at {base_dir}")
        return

    print(f"Scanning dataset at: {base_dir}\n")
    
    grand_total_img = 0
    grand_total_vid = 0
    
    # Process Live and Spoof
    categories = ['live', 'spoof']
    
    all_table_data = []

    for category in categories:
        cat_path = os.path.join(base_dir, category)
        print(f"--- Analyzing {category.upper()} ---")
        
        counts = get_category_counts(cat_path)
        
        if counts is None:
            print(f"  Directory not found: {cat_path}")
            continue
            
        cat_total_img = 0
        cat_total_vid = 0
        
        # Sort by type name
        for type_name in sorted(counts.keys()):
            n_img = counts[type_name]['images']
            n_vid = counts[type_name]['videos']
            
            cat_total_img += n_img
            cat_total_vid += n_vid
            
            all_table_data.append([category.upper(), type_name, n_img, n_vid])

        grand_total_img += cat_total_img
        grand_total_vid += cat_total_vid
        
        print(f"  Total {category.upper()}: {cat_total_img} Images, {cat_total_vid} Videos")
        print("")

    # Print clean table
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"{'CATEGORY':<10} | {'SUB-TYPE':<20} | {'IMAGES':<10} | {'VIDEOS':<10}")
    report_lines.append("-" * 60)
    for row in all_table_data:
        report_lines.append(f"{row[0]:<10} | {row[1]:<20} | {row[2]:<10} | {row[3]:<10}")
    report_lines.append("-" * 60)
    report_lines.append(f"{'TOTAL':<33} | {grand_total_img:<10} | {grand_total_vid:<10}")
    report_lines.append("="*60)
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open("dataset_stats_clean.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

if __name__ == "__main__":
    main()
