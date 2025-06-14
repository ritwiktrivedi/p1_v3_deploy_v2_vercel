# /// script
# dependencies = [
#   "requests",
#   "rich",
#   "beautifulsoup4",
#   "Pillow",
#   "html2text",
#   "openai",
#   "markdownify",
#   "semantic-text-splitter",
#   "numpy",
#   "tqdm",
#   "uvicorn",
#   "fastapi",
#   "pydantic",
#   "argparse",
# ]
# ///

import os
import re
import httpx
import html2text
from bs4 import BeautifulSoup
from PIL import Image
import base64
import mimetypes
import io
from openai import OpenAI


def get_openai_client():
    """Get OpenAI client with API key and base URL configuration"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    else:
        return OpenAI(api_key=api_key)


def get_image_description(image_path):
    """Get a description of the image using OpenAI's vision model."""
    try:
        client = get_openai_client()

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        image_base64 = base64.b64encode(image_data).decode('utf-8')
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Caption this image. Provide a brief, descriptive caption that would be useful for accessibility and document understanding."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150,
            temperature=0.3
        )

        return response.choices[0].message.content if response.choices[0].message.content else "No description available."

    except Exception as e:
        print(f"Error getting image description for {image_path}: {e}")
        return "Image description unavailable."


def process_markdown_images(md_content, md_file_path, base_dir):
    """Process images in markdown content and add descriptions"""
    # Regex to find markdown images: ![alt](src)
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    images = re.findall(img_pattern, md_content)

    if not images:
        return md_content

    print(f"  Processing {len(images)} images in markdown...")

    processed_content = md_content

    for i, (alt_text, src) in enumerate(images, 1):
        # Skip if image already has a description (check if next line starts with *)
        img_match = re.search(
            rf'!\[{re.escape(alt_text)}\]\({re.escape(src)}\)', processed_content)
        if img_match:
            # Check if there's already a description after this image
            after_img = processed_content[img_match.end():]
            if after_img.strip().startswith('*') and after_img.strip().split('\n')[0].endswith('*'):
                print(
                    f"    Skipping image {i}/{len(images)}: already has description")
                continue

        # Handle relative paths
        if not src.startswith(('http://', 'https://', '/')):
            full_img_path = os.path.join(os.path.dirname(md_file_path), src)
        else:
            if src.startswith('/'):
                full_img_path = os.path.join(base_dir, src.lstrip('/'))
            else:
                continue  # Skip external URLs

        # Check if image file exists
        if os.path.exists(full_img_path):
            try:
                print(
                    f"    Getting description for image {i}/{len(images)}: {os.path.basename(full_img_path)}")
                description = get_image_description(full_img_path)

                if description and description != "Image description unavailable.":
                    # Replace the image with image + description
                    old_img = f"![{alt_text}]({src})"
                    new_img = f"![{alt_text}]({src})\n*{description}*"
                    processed_content = processed_content.replace(
                        old_img, new_img, 1)

            except Exception as e:
                print(
                    f"    Warning: Could not process image {full_img_path}: {e}")
        else:
            print(f"    Warning: Image not found: {full_img_path}")

    return processed_content


def process_markdown_file(md_path, base_dir, output_dir):
    """Process existing markdown file and add image descriptions"""
    try:
        with open(md_path, "r", encoding="utf-8") as file:
            md_content = file.read()

        # Process images and add descriptions
        enhanced_content = process_markdown_images(
            md_content, md_path, base_dir)

        # Compute output path while preserving the relative structure
        rel_path = os.path.relpath(md_path, base_dir)
        output_md_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)

        # Write the enhanced markdown content
        with open(output_md_path, "w", encoding="utf-8") as md_file:
            md_file.write(enhanced_content)

        print(f"  ✓ Processed: {rel_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error processing {md_path}: {e}")
        return False


def convert_html_to_md(html_path, base_dir, output_dir):
    """Convert HTML file to Markdown with image descriptions (original function)"""
    try:
        with open(html_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, "html.parser")

        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = 0
        h.unicode_snob = True
        h.ignore_tables = False

        md_content = h.handle(str(soup))

        images = soup.find_all("img")
        description_images = ""

        if images:
            print(f"  Processing {len(images)} images...")

        for i, img in enumerate(images, 1):
            src = img.get("src")
            if not src:
                continue

            if not src.startswith(('http://', 'https://', '/')):
                full_img_path = os.path.join(os.path.dirname(html_path), src)
            else:
                if src.startswith('/'):
                    full_img_path = os.path.join(base_dir, src.lstrip('/'))
                else:
                    continue

            if os.path.exists(full_img_path):
                try:
                    print(
                        f"    Getting description for image {i}/{len(images)}: {os.path.basename(full_img_path)}")
                    description = get_image_description(full_img_path)
                    if description and description != "Image description unavailable.":
                        alt_text = img.get("alt", description)
                        description_images += f"\n![{alt_text}]({src})\n*{description}*\n\n"
                    else:
                        alt_text = img.get("alt", "Image")
                        description_images += f"\n![{alt_text}]({src})\n\n"
                except Exception as e:
                    print(
                        f"    Warning: Could not process image {full_img_path}: {e}")
                    alt_text = img.get("alt", "Image")
                    description_images += f"\n![{alt_text}]({src})\n\n"
            else:
                print(f"    Warning: Image not found: {full_img_path}")
                alt_text = img.get("alt", "Image")
                description_images += f"\n![{alt_text}]({src})\n\n"

        if description_images:
            md_content += "\n\n## Images\n" + description_images

        rel_path = os.path.relpath(html_path, base_dir)
        output_md_path = os.path.join(
            output_dir, rel_path).replace(".html", ".md")
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)

        with open(output_md_path, "w", encoding="utf-8") as md_file:
            md_file.write(md_content)

        print(
            f"  ✓ Converted: {rel_path} -> {os.path.relpath(output_md_path, output_dir)}")
        return True

    except Exception as e:
        print(f"  ✗ Error converting {html_path}: {e}")
        return False


if __name__ == "__main__":
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)

    # Print configuration
    vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
    base_url = os.getenv("OPENAI_BASE_URL")

    print(f"Configuration:")
    print(f"  Vision Model: {vision_model}")
    print(f"  Base URL: {base_url if base_url else 'OpenAI default'}")
    print()

    # Create output directory
    os.makedirs("./markdowns", exist_ok=True)

    base_dir = "./tools-in-data-science-public"

    if not os.path.exists(base_dir):
        print(f"ERROR: Base directory '{base_dir}' not found")
        print("Please ensure the files directory exists")
        exit(1)

    # Find both HTML and Markdown files
    files_to_process = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith((".html", ".md")):
                files_to_process.append(
                    (os.path.join(root, file), file.endswith(".html")))

    if not files_to_process:
        print(f"No HTML or Markdown files found in '{base_dir}'")
        exit(1)

    html_count = sum(1 for _, is_html in files_to_process if is_html)
    md_count = len(files_to_process) - html_count

    print(f"Found {len(files_to_process)} files to process:")
    print(f"  HTML files: {html_count}")
    print(f"  Markdown files: {md_count}")
    print("Starting processing...\n")

    successful = 0
    failed = 0

    for i, (file_path, is_html) in enumerate(files_to_process, 1):
        rel_path = os.path.relpath(file_path, base_dir)
        file_type = "HTML" if is_html else "Markdown"
        print(f"[{i}/{len(files_to_process)}] Processing {file_type}: {rel_path}")

        try:
            if is_html:
                success = convert_html_to_md(
                    file_path, base_dir, "./markdowns")
            else:
                success = process_markdown_file(
                    file_path, base_dir, "./markdowns")

            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed += 1

    print(f"\nProcessing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(files_to_process)}")

    if successful > 0:
        print(f"\nProcessed files saved to './markdowns/' directory")
        print("You can now run the embedding script to generate embeddings.")
