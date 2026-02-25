"""
MKG-Y Data Scraping Script (Label-based Format)

This script processes MKG-Y data which uses human-readable labels instead of URIs.
Example: "Marjorie_Gateson actedIn Honolulu_Lu ."

Features:
1. Parse triples with entity/relation labels
2. Extract readable text from labels (convert underscores to spaces)
3. Optionally query Wikidata for enriched descriptions
4. Download images from Wikidata

Output:
- Processed triples (numeric IDs)
- Entity and relation vocabularies
- Entity and relation text descriptions
- Downloaded images (cached)
- Metadata

Usage:
    python scrape_mkg_data_labels.py --help
"""
import argparse
import json
import time
import re
import requests
from pathlib import Path
from tqdm.auto import tqdm
from io import BytesIO
from PIL import Image


# ============================================================================
# TRIPLE PARSING
# ============================================================================

def parse_triples(file_path):
    """Parse MKG-Y triples with label format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    lines = content.split('\n')
    triples = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Remove trailing dot
        line = line[:-2]
        parts = line.split(' ')
        
        if len(parts) < 3:
            print(f"Warning: Skipping invalid triple: {line}")
            continue
        
        # Handle cases where entity names might have spaces (shouldn't happen but just in case)
        if len(parts) == 3:
            h, r, t = parts
        else:
            # Take first as head, last as tail, middle as relation
            h = parts[0]
            t = parts[-1]
            r = ' '.join(parts[1:-1])
        
        triples.append((h, r, t))
    
    return triples


# ============================================================================
# TEXT EXTRACTION FROM LABELS
# ============================================================================

def clean_label(label):
    """
    Convert entity/relation label to readable text.
    
    Examples:
        "Marjorie_Gateson" → "Marjorie Gateson"
        "Count_Max_(1991_film)" → "Count Max"
        "actedIn" → "acted in"
        "wasBornIn" → "was born in"
        "FC_Jūrmala" → "FC Jūrmala"
    """
    original_label = label
    
    # Remove disambiguation text in parentheses
    label = re.sub(r'\([^)]*\)', '', label).strip()
    
    # Replace underscores with spaces
    label = label.replace('_', ' ')
    
    # Handle camelCase: insert space before capitals (for relations)
    # But don't split acronyms like "FC" or "USA"
    label = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', label)
    
    # Clean up multiple spaces
    label = re.sub(r'\s+', ' ', label).strip()
    
    # If original started with lowercase (likely a relation), lowercase all words
    if original_label and original_label[0].islower():
        label = label.lower()
    
    return label if label else original_label


def search_wikidata_for_entity(label, timeout=10, rate_limit_delay=0.3, max_retries=3):
    """
    Search Wikidata for an entity by label and return Q-number and description.
    Returns: (q_number, description) or (None, None)
    """
    # Clean label for search
    search_label = clean_label(label)
    
    # Use Wikidata's search API
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'type': 'item',
        'search': search_label,
        'limit': 1
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            # Rate limiting before each request
            time.sleep(rate_limit_delay)
            
            response = requests.get(search_url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            
            # Check if we got results
            if data.get('search') and len(data['search']) > 0:
                result = data['search'][0]
                q_number = result.get('id')
                description = result.get('description', '')
                label_text = result.get('label', search_label)
                
                if description:
                    return q_number, f"{label_text} ({description})"
                else:
                    return q_number, label_text
            break  # No results found, don't retry
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:  # Rate limited
                if attempt < max_retries - 1:
                    wait_time = rate_limit_delay * (2 ** (attempt + 1))  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed, suppress verbose error
                    pass
            else:
                break  # Other HTTP errors, don't retry
        except requests.exceptions.RequestException:
            break  # Connection errors, don't retry
        except (KeyError, ValueError, IndexError):
            break  # Parse errors, don't retry
    
    return None, None


def extract_text_from_label(label, use_wikidata=False, is_relation=False, timeout=10):
    """
    Extract readable text from entity/relation label.
    
    Args:
        label: Entity or relation label (e.g., "Marjorie_Gateson", "actedIn")
        use_wikidata: If True, try to query Wikidata for description
        is_relation: If True, this is a relation (don't query Wikidata)
        timeout: Timeout for Wikidata queries
    
    Returns:
        tuple: (text, found_in_wikidata) where found_in_wikidata is True if Wikidata match found
    """
    # Always get clean label as fallback
    clean_text = clean_label(label)
    
    # For relations, just return clean label
    if is_relation:
        return clean_text, False
    
    # For entities, optionally query Wikidata
    if use_wikidata:
        q_number, description = search_wikidata_for_entity(label, timeout)
        if description:
            return description, True
        else:
            print(f"No Wikidata description found for '{label}', using cleaned label")
            return clean_text, False
    
    return clean_text, False


# ============================================================================
# IMAGE DOWNLOADING
# ============================================================================

def get_wikidata_qnumber(label, timeout=10, rate_limit_delay=0.3):
    """Get Wikidata Q-number for an entity label."""
    search_label = clean_label(label)
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'type': 'item',
        'search': search_label,
        'limit': 1
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # Rate limiting
        time.sleep(rate_limit_delay)
        
        response = requests.get(search_url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        if data.get('search') and len(data['search']) > 0:
            q_number = data['search'][0].get('id')
            return q_number
    except Exception as e:
        # Silent failure for image downloads (too verbose otherwise)
        pass
    
    return None


def query_wikidata_for_image(label, timeout=10):
    """Query Wikidata SPARQL endpoint for entity image URL using label search."""
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
    except ImportError:
        return None
    
    # First, get Q-number for this label
    q_number = get_wikidata_qnumber(label, timeout)
    if not q_number:
        return None
    
    entity_uri = f"http://www.wikidata.org/entity/{q_number}"
    return query_wikidata_sparql_for_image(entity_uri, timeout)


def query_wikidata_sparql_for_image(entity_uri, timeout=10):
    """Query Wikidata SPARQL endpoint for entity image URL given entity URI."""
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
    except ImportError:
        return None
    
    query = f"""
    SELECT ?image WHERE {{
        <{entity_uri}> wdt:P18 ?image .
    }}
    LIMIT 1
    """
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setTimeout(timeout)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        time.sleep(0.1)  # Rate limiting
        results = sparql.query().convert()
        
        if results and "results" in results and "bindings" in results["results"]:
            bindings = results["results"]["bindings"]
            if bindings and "image" in bindings[0]:
                return bindings[0]["image"]["value"]
    except Exception:
        pass
    
    return None


def download_and_save_image(image_url, save_path, max_size_mb=5):
    """Download image from URL and save to disk."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        }
        response = requests.get(image_url, timeout=10, stream=True, verify=True, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            return False
        
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            return False
        
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img.save(save_path, 'JPEG', quality=85)
        
        return True
    except Exception:
        return False


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Scrape MKG-Y data from label-based triple files')
    
    # Input/Output paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to raw MKG-Y data directory (containing train.txt, valid.txt, test.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--dataset_name', type=str, default='MKG-Y',
                        help='Dataset name')
    
    # Text extraction options
    parser.add_argument('--text_mode', type=str, default='label_only',
                        choices=['label_only', 'wikidata_search'],
                        help='Text extraction method: label_only (fast) or wikidata_search (slow, detailed)')
    parser.add_argument('--text_sample', type=int, default=None,
                        help='Sample only N entities for Wikidata queries (for testing)')
    parser.add_argument('--rate_limit', type=float, default=0.3,
                        help='Delay between Wikidata API requests in seconds (default: 0.3, ~3 req/sec)')
    
    # Image downloading options
    parser.add_argument('--download_images', action='store_true',
                        help='Download images from Wikidata (downloads all by default)')
    parser.add_argument('--image_sample', type=int, default=None,
                        help='Limit to first N entities for image downloads (default: None = all entities)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to successfully download (default: None = no limit)')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"{args.dataset_name} DATA SCRAPING (LABEL FORMAT)")
    print("="*70)
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Text mode: {args.text_mode}")
    if args.text_mode == 'wikidata_search':
        print(f"Rate limit: {args.rate_limit}s delay (~{1/args.rate_limit:.1f} req/sec)")
    print(f"Download images: {args.download_images}")
    if args.download_images:
        print(f"Image rate limit: {args.rate_limit}s delay (~{1/args.rate_limit:.1f} req/sec)")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load triples
    # ========================================================================
    print("\n[1/5] Loading triples...")
    train_triples = parse_triples(data_dir / 'train.txt')
    valid_triples = parse_triples(data_dir / 'valid.txt')
    test_triples = parse_triples(data_dir / 'test.txt')
    
    print(f"  Train: {len(train_triples):,}")
    print(f"  Valid: {len(valid_triples):,}")
    print(f"  Test: {len(test_triples):,}")
    
    # ========================================================================
    # STEP 2: Create vocabularies
    # ========================================================================
    print("\n[2/5] Creating vocabularies...")
    entities_set = set()
    relations_set = set()
    
    for h, r, t in train_triples + valid_triples + test_triples:
        entities_set.add(h)
        entities_set.add(t)
        relations_set.add(r)
    
    entities = sorted(list(entities_set))
    relations = sorted(list(relations_set))
    
    entity2id = {ent: idx for idx, ent in enumerate(entities)}
    relation2id = {rel: idx for idx, rel in enumerate(relations)}
    
    print(f"  Entities: {len(entities):,}")
    print(f"  Relations: {len(relations):,}")
    
    # ========================================================================
    # STEP 3: Extract entity and relation texts
    # ========================================================================
    print("\n[3/5] Extracting texts...")
    
    use_wikidata = (args.text_mode == 'wikidata_search')
    
    # Extract relation texts (always simple conversion)
    print("  Extracting relation texts...")
    relation_texts = []
    for relation_label in tqdm(relations, desc="  Relations"):
        text, _ = extract_text_from_label(relation_label, use_wikidata=False, is_relation=True)
        relation_texts.append(text)
    
    # Extract entity texts
    entity_texts = []
    wikidata_found_count = 0
    
    if use_wikidata:
        print(f"  Method: Wikidata search (slow)")
        sample_size = args.text_sample if args.text_sample else len(entities)
        print(f"  Processing {sample_size:,} entities with Wikidata...")
        print(f"  Rate limit: {args.rate_limit}s delay (~{1/args.rate_limit:.1f} requests/sec)")
        
        for entity_label in tqdm(entities[:sample_size], desc="  Entities (Wikidata)"):
            # Pass rate limit to search function
            q_number, description = search_wikidata_for_entity(entity_label, timeout=10, rate_limit_delay=args.rate_limit)
            if description:
                entity_texts.append(description)
                wikidata_found_count += 1
            else:
                # Fallback to cleaned label
                text, _ = extract_text_from_label(entity_label, use_wikidata=False, is_relation=False)
                entity_texts.append(text)
        
        # Use simple extraction for remaining
        if sample_size < len(entities):
            print(f"  Processing remaining {len(entities) - sample_size:,} entities with labels...")
            for entity_label in tqdm(entities[sample_size:], desc="  Entities (labels)"):
                text, _ = extract_text_from_label(entity_label, use_wikidata=False, is_relation=False)
                entity_texts.append(text)
    else:
        print("  Method: Label cleaning (fast)")
        for entity_label in tqdm(entities, desc="  Entities"):
            text, _ = extract_text_from_label(entity_label, use_wikidata=False, is_relation=False)
            entity_texts.append(text)
    
    print(f"  ✓ Extracted {len(entity_texts):,} entity texts")
    print(f"  ✓ Extracted {len(relation_texts):,} relation texts")
    if use_wikidata and sample_size > 0:
        print(f"  ✓ Found Wikidata matches: {wikidata_found_count:,}/{sample_size:,} ({wikidata_found_count/sample_size*100:.1f}%)")
    
    # Show some examples
    print("\n  Examples:")
    for i in range(min(5, len(entities))):
        print(f"    {entities[i]} → {entity_texts[i]}")
    for i in range(min(3, len(relations))):
        print(f"    {relations[i]} → {relation_texts[i]}")
    
    # ========================================================================
    # STEP 4: Download images (optional)
    # ========================================================================
    image_info = {}
    image_urls = {}
    
    if args.download_images:
        print("\n[4/5] Downloading images from Wikidata...")
        
        images_dir = output_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        # Determine which entities to process
        if args.image_sample:
            entities_to_process = entities[:args.image_sample]
            print(f"  Processing sample of {len(entities_to_process):,} entities...")
        else:
            entities_to_process = entities
            print(f"  Processing all {len(entities_to_process):,} entities...")
        
        # Phase 1: Query all image URLs
        print(f"  Phase 1: Querying image URLs...")
        print(f"  Rate limit: {args.rate_limit}s delay per request")
        for entity_label in tqdm(entities_to_process, desc="  Querying URLs"):
            entity_id = entity2id[entity_label]
            # Pass rate limit to image query
            q_number = get_wikidata_qnumber(entity_label, timeout=10, rate_limit_delay=args.rate_limit)
            if q_number:
                entity_uri = f"http://www.wikidata.org/entity/{q_number}"
                image_url = query_wikidata_sparql_for_image(entity_uri)
                if image_url:
                    image_urls[entity_id] = image_url
        
        print(f"  ✓ Found {len(image_urls):,} image URLs")
        
        # Save URLs immediately as backup
        with open(output_dir / 'image_urls.json', 'w', encoding='utf-8') as f:
            json.dump(image_urls, f, indent=2)
        print(f"  ✓ Saved URLs to image_urls.json")
        
        # Phase 2: Download images from saved URLs
        print(f"  Phase 2: Downloading images...")
        downloaded_count = 0
        items_to_download = list(image_urls.items())
        
        # Apply max_images limit if specified
        if args.max_images:
            items_to_download = items_to_download[:args.max_images]
        
        for entity_id, image_url in tqdm(items_to_download, desc="  Downloading"):
            image_path = images_dir / f"entity_{entity_id}.jpg"
            
            # Skip if already downloaded
            if image_path.exists():
                image_info[entity_id] = f"images/entity_{entity_id}.jpg"
                downloaded_count += 1
                continue
            
            # Download image
            if download_and_save_image(image_url, image_path):
                image_info[entity_id] = f"images/entity_{entity_id}.jpg"
                downloaded_count += 1
        
        print(f"  ✓ Downloaded {downloaded_count:,} images")
        print(f"  ✓ Coverage: {downloaded_count/len(entities)*100:.1f}%")
    else:
        print("\n[4/5] Skipping image download")
    
    # ========================================================================
    # STEP 5: Save everything
    # ========================================================================
    print("\n[5/5] Saving data...")
    
    # Save triples (original labels)
    with open(output_dir / 'triples_train.txt', 'w', encoding='utf-8') as f:
        for h, r, t in train_triples:
            f.write(f"{h} {r} {t}\n")
    
    with open(output_dir / 'triples_valid.txt', 'w', encoding='utf-8') as f:
        for h, r, t in valid_triples:
            f.write(f"{h} {r} {t}\n")
    
    with open(output_dir / 'triples_test.txt', 'w', encoding='utf-8') as f:
        for h, r, t in test_triples:
            f.write(f"{h} {r} {t}\n")
    
    # Save vocabularies
    with open(output_dir / 'entity2id.json', 'w', encoding='utf-8') as f:
        json.dump(entity2id, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'relation2id.json', 'w', encoding='utf-8') as f:
        json.dump(relation2id, f, indent=2, ensure_ascii=False)
    
    # Save texts
    with open(output_dir / 'entity_texts.json', 'w', encoding='utf-8') as f:
        json.dump(entity_texts, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'relation_texts.json', 'w', encoding='utf-8') as f:
        json.dump(relation_texts, f, indent=2, ensure_ascii=False)
    
    # Save image info
    with open(output_dir / 'image_paths.json', 'w', encoding='utf-8') as f:
        json.dump(image_info, f, indent=2)
    
    # Save metadata
    metadata = {
        'dataset': args.dataset_name,
        'format': 'label-based',
        'num_entities': len(entities),
        'num_relations': len(relations),
        'num_train_triples': len(train_triples),
        'num_valid_triples': len(valid_triples),
        'num_test_triples': len(test_triples),
        'text_extraction_mode': args.text_mode,
        'images_downloaded': args.download_images,
        'num_image_urls': len(image_urls),
        'num_images': len(image_info),
        'image_coverage': len(image_info) / len(entities) if entities else 0.0,
    }
    
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved to {output_dir}")
    
    print("\n" + "="*70)
    print("✅ DATA SCRAPING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  triples_train.txt, triples_valid.txt, triples_test.txt")
    print("  entity2id.json, relation2id.json")
    print("  entity_texts.json, relation_texts.json")
    print("  image_paths.json")
    if args.download_images:
        print(f"  images/ ({len(image_info)} images)")
    print("  metadata.json")
    print("\nNext step: Run prepare_embeddings.ipynb to generate embeddings")


if __name__ == '__main__':
    main()
