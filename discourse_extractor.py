# /// script
# dependencies = [
#   "requests",
#   "beautifulsoup4",
#   "html2text",
#   "pillow",
#   "openai",  # pip install openai
#   "anthropic",  # pip install anthropic
#   "google-generativeai",  # pip install google-generativeai
#   "transformers",  # pip install transformers torch (for local BLIP)
#   "torch",  # pip install torch (for local BLIP)
# ]
# ///

import requests
import json
import os
import re
from pathlib import Path
from datetime import datetime
import html2text
import base64
from PIL import Image
import io
from urllib.parse import urljoin, urlparse
import time


def load_cookies():
    """Load cookies from cookies.txt file"""
    try:
        with open("cookies.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Error: cookies.txt file not found!")
        return None


def fetch_discourse_data(cookies):
    """Fetch data from Discourse API"""
    headers = {
        "cookie": cookies,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # Search URL - you can modify the search parameters as needed
    search_url = "https://discourse.onlinedegree.iitm.ac.in/search.json"
    params = {
        "q": "#courses:tds-kb after:2025-01-01 before:2025-04-15 order:latest"
    }

    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def get_topic_details(topic_id, cookies):
    """Get detailed topic information including posts"""
    headers = {
        "cookie": cookies,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    topic_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}.json"

    try:
        response = requests.get(topic_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching topic {topic_id}: {e}")
        return None


class ImageProcessor:
    """Handle image downloading and description generation"""

    def __init__(self, cookies, base_url="https://discourse.onlinedegree.iitm.ac.in"):
        self.cookies = cookies
        self.base_url = base_url
        self.headers = {
            "cookie": cookies,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.image_cache = {}
        self.description_cache = {}

        # Configure your LLM API (choose one)
        self.llm_provider = self._setup_llm()

    def _setup_llm(self):
        """Setup LLM provider for image descriptions"""
        # Option 1: OpenAI
        if os.getenv('OPENAI_API_KEY'):
            try:
                import openai
                client = openai.OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    # Optionally set base_url if using a different endpoint
                    # base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
                )
                # Test the connection
                client.models.list()
                print("✓ OpenAI API configured successfully")
                return {
                    'type': 'openai',
                    'client': client
                }
            except Exception as e:
                print(f"⚠ OpenAI setup failed: {e}")
                print("Falling back to other providers...")

        # Option 2: Google Gemini
        if os.getenv('GOOGLE_API_KEY'):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

                # Test the connection by creating a model instance
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                print("✓ Google Gemini API configured successfully")
                return {
                    'type': 'gemini',
                    'model': model
                }
            except Exception as e:
                print(f"⚠ Gemini setup failed: {e}")
                print("Falling back to other providers...")

        # Option 3: Anthropic Claude
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                import anthropic
                client = anthropic.Anthropic(
                    api_key=os.getenv('ANTHROPIC_API_KEY'))
                print("✓ Anthropic API configured successfully")
                return {
                    'type': 'anthropic',
                    'client': client
                }
            except Exception as e:
                print(f"⚠ Anthropic setup failed: {e}")

        # Option 4: Local LLM (like Ollama with llava)
        if os.getenv('OLLAMA_HOST'):
            try:
                host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
                # Test connection
                test_response = requests.get(f"{host}/api/tags", timeout=5)
                if test_response.status_code == 200:
                    print("✓ Ollama API configured successfully")
                    return {
                        'type': 'ollama',
                        'host': host
                    }
                else:
                    print(f"⚠ Ollama not responding at {host}")
            except Exception as e:
                print(f"⚠ Ollama setup failed: {e}")

        # Option 5: Hugging Face Transformers (local processing)
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch

            print("✓ Setting up local BLIP model for image descriptions...")
            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base")

            return {
                'type': 'huggingface_blip',
                'processor': processor,
                'model': model
            }
        except ImportError:
            print("⚠ Hugging Face transformers not available")
        except Exception as e:
            print(f"⚠ Local BLIP model setup failed: {e}")

        print("⚠ No LLM API configured. Images will be included without descriptions.")
        print("Available options:")
        print("  • Set OPENAI_API_KEY for GPT-4 Vision")
        print("  • Set GOOGLE_API_KEY for Gemini 2.0 Flash")
        print("  • Set ANTHROPIC_API_KEY for Claude 3")
        print("  • Set OLLAMA_HOST for local LLaVA")
        print("  • Install transformers: pip install transformers torch pillow")
        return None

    def download_image(self, image_url, images_dir="images"):
        """Download image from URL"""
        try:
            # Handle relative URLs
            if not image_url.startswith('http'):
                image_url = urljoin(self.base_url, image_url)

            # Check cache
            if image_url in self.image_cache:
                return self.image_cache[image_url]

            # Create images directory
            Path(images_dir).mkdir(exist_ok=True)

            # Download image
            response = requests.get(image_url, headers=self.headers)
            response.raise_for_status()

            # Generate filename
            parsed_url = urlparse(image_url)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                filename = f"image_{hash(image_url) % 10000}.jpg"

            filepath = Path(images_dir) / filename

            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)

            # Cache result
            result = {
                'local_path': str(filepath),
                'original_url': image_url,
                'filename': filename
            }
            self.image_cache[image_url] = result

            print(f"Downloaded: {filename}")
            return result

        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
            return None

    def generate_image_description(self, image_path):
        """Generate description using LLM"""
        if not self.llm_provider:
            return f"Image: {os.path.basename(image_path)}"

        # Check cache
        if image_path in self.description_cache:
            return self.description_cache[image_path]

        try:
            if self.llm_provider['type'] == 'openai':
                description = self._describe_with_openai(image_path)
            elif self.llm_provider['type'] == 'gemini':
                description = self._describe_with_gemini(image_path)
            elif self.llm_provider['type'] == 'anthropic':
                description = self._describe_with_anthropic(image_path)
            elif self.llm_provider['type'] == 'ollama':
                description = self._describe_with_ollama(image_path)
            elif self.llm_provider['type'] == 'huggingface_blip':
                description = self._describe_with_blip(image_path)
            else:
                description = f"Image: {os.path.basename(image_path)}"

            # Cache result
            self.description_cache[image_path] = description
            return description

        except Exception as e:
            print(f"Error generating description for {image_path}: {e}")
            fallback_desc = f"Image: {os.path.basename(image_path)}"
            self.description_cache[image_path] = fallback_desc
            return fallback_desc

    def _describe_with_openai(self, image_path):
        """Generate description using OpenAI GPT-4 Vision"""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(
                    image_file.read()).decode('utf-8')

            # Determine image type
            image_type = "jpeg"
            if image_path.lower().endswith('.png'):
                image_type = "png"
            elif image_path.lower().endswith('.gif'):
                image_type = "gif"
            elif image_path.lower().endswith('.webp'):
                image_type = "webp"

            response = self.llm_provider['client'].chat.completions.create(
                model="gpt-4o-mini",  # Updated model name
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please provide a detailed description of this image. Focus on the main content, text if visible, charts/diagrams if present, and any educational or technical content. Be concise but comprehensive."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_type};base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Try Gemini as fallback if available
            if os.getenv('GOOGLE_API_KEY') and self.llm_provider['type'] == 'openai':
                print("Trying Gemini as fallback...")
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                    model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    return self._describe_with_gemini_fallback(image_path, model)
                except Exception as gemini_e:
                    print(f"Gemini fallback also failed: {gemini_e}")

            # Fallback to basic description
            return f"Image file: {os.path.basename(image_path)}"

    def _describe_with_gemini(self, image_path):
        """Generate description using Google Gemini"""
        try:
            from PIL import Image as PILImage

            # Load image
            img = PILImage.open(image_path)

            # Prepare the prompt
            prompt = """Please provide a detailed description of this image. Focus on the main content, text if visible, charts/diagrams if present, and any educational or technical content. Be concise but comprehensive."""

            # Generate description
            response = self.llm_provider['model'].generate_content([
                                                                   prompt, img])

            return response.text

        except Exception as e:
            print(f"Gemini API error: {e}")
            # Try OpenAI as fallback if available
            if os.getenv('OPENAI_API_KEY') and self.llm_provider['type'] == 'gemini':
                print("Trying OpenAI as fallback...")
                try:
                    import openai
                    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                    return self._describe_with_openai_fallback(image_path, client)
                except Exception as openai_e:
                    print(f"OpenAI fallback also failed: {openai_e}")

            return f"Image file: {os.path.basename(image_path)}"

    def _describe_with_gemini_fallback(self, image_path, model):
        """Fallback method for Gemini when called from OpenAI failure"""
        try:
            from PIL import Image as PILImage

            img = PILImage.open(image_path)
            prompt = "Please provide a detailed description of this image. Focus on the main content, text if visible, charts/diagrams if present, and any educational or technical content. Be concise but comprehensive."

            response = model.generate_content([prompt, img])
            return response.text

        except Exception as e:
            print(f"Gemini fallback error: {e}")
            return f"Image file: {os.path.basename(image_path)}"

    def _describe_with_openai_fallback(self, image_path, client):
        """Fallback method for OpenAI when called from Gemini failure"""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(
                    image_file.read()).decode('utf-8')

            image_type = "jpeg"
            if image_path.lower().endswith('.png'):
                image_type = "png"
            elif image_path.lower().endswith('.gif'):
                image_type = "gif"
            elif image_path.lower().endswith('.webp'):
                image_type = "webp"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please provide a detailed description of this image. Focus on the main content, text if visible, charts/diagrams if present, and any educational or technical content. Be concise but comprehensive."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_type};base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI fallback error: {e}")
            return f"Image file: {os.path.basename(image_path)}"

    def _describe_with_blip(self, image_path):
        """Generate description using local BLIP model"""
        from PIL import Image as PILImage

        # Load and process image
        image = PILImage.open(image_path).convert('RGB')
        inputs = self.llm_provider['processor'](image, return_tensors="pt")

        # Generate description
        with torch.no_grad():
            out = self.llm_provider['model'].generate(**inputs, max_length=50)

        description = self.llm_provider['processor'].decode(
            out[0], skip_special_tokens=True)
        return description

    def _describe_with_anthropic(self, image_path):
        """Generate description using Anthropic Claude"""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(
                    image_file.read()).decode('utf-8')

            # Get image type
            image_type = "image/jpeg"
            if image_path.lower().endswith('.png'):
                image_type = "image/png"
            elif image_path.lower().endswith('.gif'):
                image_type = "image/gif"
            elif image_path.lower().endswith('.webp'):
                image_type = "image/webp"

            response = self.llm_provider['client'].messages.create(
                model="claude-3-5-sonnet-20241022",  # Updated model
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_type,
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": "Please provide a detailed description of this image. Focus on the main content, text if visible, charts/diagrams if present, and any educational or technical content. Be concise but comprehensive."
                            }
                        ]
                    }
                ]
            )

            return response.content[0].text

        except Exception as e:
            print(f"Anthropic API error: {e}")
            return f"Image file: {os.path.basename(image_path)}"

    def _describe_with_ollama(self, image_path):
        """Generate description using local Ollama with LLaVA"""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(
                    image_file.read()).decode('utf-8')

            response = requests.post(
                f"{self.llm_provider['host']}/api/generate",
                json={
                    "model": "llava",  # or llava:13b, llava:34b
                    "prompt": "Please provide a detailed description of this image. Focus on the main content, text if visible, charts/diagrams if present, and any educational or technical content. Be concise but comprehensive.",
                    "images": [base64_image],
                    "stream": False
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json()['response']
            else:
                raise Exception(
                    f"Ollama API error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Ollama API error: {e}")
            return f"Image file: {os.path.basename(image_path)}"


def should_process_image(img_tag, img_src):
    """
    Determine if an image should be processed based on context and characteristics.
    Returns True for content images, False for UI elements.
    """

    # Skip if no src
    if not img_src:
        return False

    # URL-based filtering - skip common UI image paths
    ui_patterns = [
        '/avatars/',
        '/avatar/',
        '/user_avatar/',
        '/badges/',
        '/emoji/',
        '/plugins/',
        '/assets/',
        '/stylesheets/',
        '/icons/',
        '/favicon',
        '/logo',
        '.svg',  # Many UI elements are SVG
    ]

    img_src_lower = img_src.lower()
    if any(pattern in img_src_lower for pattern in ui_patterns):
        return False

    # Class-based filtering - skip images with UI-related CSS classes
    img_classes = img_tag.get('class', [])
    if isinstance(img_classes, str):
        img_classes = img_classes.split()

    ui_classes = [
        'avatar',
        'emoji',
        'badge',
        'icon',
        'logo',
        'user-avatar',
        'topic-avatar',
        'favicon',
    ]

    if any(ui_class in ' '.join(img_classes).lower() for ui_class in ui_classes):
        return False

    # Size-based filtering - skip very small images (likely icons/emoji)
    width = img_tag.get('width')
    height = img_tag.get('height')

    try:
        if width and int(width) < 32:  # Less than 32px wide
            return False
        if height and int(height) < 32:  # Less than 32px tall
            return False
    except (ValueError, TypeError):
        pass

    # Alt text filtering - skip images with UI-related alt text
    alt_text = (img_tag.get('alt', '') or '').lower()
    ui_alt_patterns = [
        'avatar',
        'emoji',
        'badge',
        'icon',
        'logo',
        'profile',
        'user',
        ':',  # Emoji often have alt text like ":smile:"
    ]

    if any(pattern in alt_text for pattern in ui_alt_patterns):
        return False

    # Context-based filtering - check parent elements
    parent = img_tag.parent
    if parent:
        parent_classes = parent.get('class', [])
        if isinstance(parent_classes, str):
            parent_classes = parent_classes.split()

        ui_parent_classes = [
            'avatar',
            'user-info',
            'topic-meta',
            'post-meta',
            'emoji',
            'badge-wrapper',
        ]

        if any(ui_class in ' '.join(parent_classes).lower() for ui_class in ui_parent_classes):
            return False

    # If none of the filters caught it, assume it's content
    return True


def html_to_markdown_with_images(html_content, image_processor, images_dir="images"):
    """
    Enhanced version with better image filtering
    """
    from bs4 import BeautifulSoup

    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all images
    images = soup.find_all('img')

    processed_count = 0
    skipped_count = 0

    for img in images:
        img_src = img.get('src')

        # Apply filtering logic
        if not should_process_image(img, img_src):
            print(f"Skipping UI image: {img_src}")
            skipped_count += 1
            continue

        print(f"Processing content image: {img_src}")
        processed_count += 1

        # Download image
        image_info = image_processor.download_image(img_src, images_dir)
        if not image_info:
            continue

        # Generate description with enhanced context
        description = image_processor.generate_image_description(
            image_info['local_path'])

        # Create enhanced alt text
        original_alt = img.get('alt', '')
        enhanced_alt = f"{original_alt} - {description}".strip(' - ')

        # Update the img tag with enhanced information
        img['alt'] = enhanced_alt
        img['src'] = image_info['local_path']  # Use local path

        # Add description as a caption
        caption = soup.new_tag('p')
        caption.string = f"*Image: {description}*"
        img.insert_after(caption)

    print(
        f"Image processing summary: {processed_count} processed, {skipped_count} skipped")

    # Convert to markdown
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0

    return h.handle(str(soup))


def sanitize_filename(filename):
    """Sanitize filename for safe file creation"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = re.sub(r'\s+', '_', filename)  # Replace spaces with underscores
    return filename[:100]  # Limit length


def save_discourse_json(data, filename="discourse_data.json"):
    """Save raw discourse data as JSON"""
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print(f"Saved raw data to {filename}")


def convert_to_markdown(discourse_data, cookies, output_dir="discourse_markdowns"):
    """Convert discourse posts to markdown files with enhanced image handling"""
    if not discourse_data or 'posts' not in discourse_data:
        print("No posts found in discourse data")
        return []

    # Create output directory and images subdirectory
    Path(output_dir).mkdir(exist_ok=True)
    images_dir = Path(output_dir) / "images"

    # Initialize image processor
    image_processor = ImageProcessor(cookies)

    markdown_files = []

    # Group posts by topic
    topics_processed = set()

    for post in discourse_data.get('posts', []):
        topic_id = post.get('topic_id')

        # Skip if we've already processed this topic
        if topic_id in topics_processed:
            continue

        topics_processed.add(topic_id)

        # Get full topic details
        topic_data = get_topic_details(topic_id, cookies)
        if not topic_data:
            continue

        # Extract topic information
        topic_title = topic_data.get('title', f'Topic_{topic_id}')
        topic_slug = topic_data.get('slug', '')

        # Create markdown content
        markdown_content = f"# {topic_title}\n\n"
        markdown_content += f"**Topic ID:** {topic_id}\n"
        markdown_content += f"**URL:** https://discourse.onlinedegree.iitm.ac.in/t/{topic_slug}/{topic_id}\n"
        markdown_content += f"**Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Add topic details
        if 'details' in topic_data:
            details = topic_data['details']
            if 'created_at' in details:
                markdown_content += f"**Created:** {details['created_at']}\n"
            if 'last_posted_at' in details:
                markdown_content += f"**Last Posted:** {details['last_posted_at']}\n"
            markdown_content += "\n"

        # Add posts
        markdown_content += "## Posts\n\n"

        for i, topic_post in enumerate(topic_data.get('post_stream', {}).get('posts', [])):
            post_content = topic_post.get('cooked', '')  # HTML content
            username = topic_post.get('username', 'Unknown')
            created_at = topic_post.get('created_at', '')

            markdown_content += f"### Post {i+1} - {username}\n"
            markdown_content += f"**Posted:** {created_at}\n\n"

            # Convert HTML to markdown with enhanced image handling
            if post_content:
                print(f"Processing post {i+1} content...")
                post_markdown = html_to_markdown_with_images(
                    post_content,
                    image_processor,
                    str(images_dir)
                )
                markdown_content += post_markdown + "\n\n"

            markdown_content += "---\n\n"

        # Save markdown file
        safe_title = sanitize_filename(topic_title)
        filename = f"{safe_title}_{topic_id}.md"
        filepath = Path(output_dir) / filename

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(markdown_content)

        markdown_files.append(str(filepath))
        print(f"Created: {filepath}")

        # Small delay to be respectful to APIs
        time.sleep(0.5)

    # Create image index
    if images_dir.exists() and list(images_dir.glob("*")):
        create_image_index(images_dir, image_processor.description_cache)

    return markdown_files


def create_image_index(images_dir, description_cache):
    """Create an index of all downloaded images with descriptions"""
    index_content = "# Image Index\n\n"
    index_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    image_files = list(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower(
    ) in ['.jpg', '.jpeg', '.png', '.gif', '.webp']]

    for img_file in sorted(image_files):
        img_path = str(img_file)
        description = description_cache.get(
            img_path, "No description available")

        index_content += f"## {img_file.name}\n\n"
        index_content += f"![{description}]({img_file.name})\n\n"
        index_content += f"**Description:** {description}\n\n"
        index_content += "---\n\n"

    # Save image index
    index_path = images_dir / "image_index.md"
    with open(index_path, "w", encoding="utf-8") as file:
        file.write(index_content)

    print(f"Created image index: {index_path}")


def combine_with_existing_markdowns(new_markdowns, existing_dir="markdowns", combined_dir="combined_markdowns"):
    """Combine new discourse markdowns with existing markdown folder"""
    combined_path = Path(combined_dir)
    combined_path.mkdir(exist_ok=True)

    # Copy existing markdowns
    existing_path = Path(existing_dir)
    if existing_path.exists():
        for md_file in existing_path.glob("*.md"):
            dest_file = combined_path / md_file.name
            with open(md_file, "r", encoding="utf-8") as src:
                content = src.read()
            with open(dest_file, "w", encoding="utf-8") as dst:
                dst.write(content)
            print(f"Copied: {md_file.name}")

    # Copy new discourse markdowns
    for md_file_path in new_markdowns:
        md_file = Path(md_file_path)
        dest_file = combined_path / f"discourse_{md_file.name}"
        with open(md_file, "r", encoding="utf-8") as src:
            content = src.read()
        with open(dest_file, "w", encoding="utf-8") as dst:
            dst.write(content)
        print(f"Added discourse file: discourse_{md_file.name}")

    print(f"\nAll markdowns combined in: {combined_dir}")
    return str(combined_path)


def create_index_file(combined_dir="combined_markdowns"):
    """Create an index file listing all markdown files"""
    combined_path = Path(combined_dir)
    index_content = "# Markdown Files Index\n\n"
    index_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # List all markdown files
    md_files = list(combined_path.glob("*.md"))

    index_content += "## Discourse Files\n\n"
    discourse_files = [f for f in md_files if f.name.startswith("discourse_")]
    for md_file in sorted(discourse_files):
        index_content += f"- [{md_file.stem}]({md_file.name})\n"

    index_content += "\n## Other Files\n\n"
    other_files = [f for f in md_files if not f.name.startswith("discourse_")]
    for md_file in sorted(other_files):
        index_content += f"- [{md_file.stem}]({md_file.name})\n"

    # Save index file
    index_path = combined_path / "README.md"
    with open(index_path, "w", encoding="utf-8") as file:
        file.write(index_content)

    print(f"Created index file: {index_path}")


def main():
    """Main execution function"""
    print("Starting Discourse to Markdown extraction with AI image descriptions...")

    # Check for required environment variables
    api_configured = any([
        os.getenv('OPENAI_API_KEY'),
        os.getenv('GOOGLE_API_KEY'),
        os.getenv('ANTHROPIC_API_KEY'),
        os.getenv('OLLAMA_HOST')
    ])

    if not api_configured:
        print("\n" + "="*60)
        print("IMAGE DESCRIPTION SETUP")
        print("="*60)
        print(
            "For AI-generated image descriptions, set one of these environment variables:")
        print("• OPENAI_API_KEY - for GPT-4 Vision")
        print("• GOOGLE_API_KEY - for Gemini 2.0 Flash")
        print("• ANTHROPIC_API_KEY - for Claude 3")
        print("• OLLAMA_HOST - for local LLaVA (e.g., http://localhost:11434)")
        print("\nImages will still be downloaded but won't have AI descriptions.")
        print("="*60 + "\n")

    # Load cookies
    cookies = load_cookies()
    if not cookies:
        return

    # Fetch discourse data
    print("Fetching discourse data...")
    discourse_data = fetch_discourse_data(cookies)
    if not discourse_data:
        return

    # Save raw JSON data
    save_discourse_json(discourse_data)

    # Convert to markdown with image processing
    print("Converting to markdown with image processing...")
    markdown_files = convert_to_markdown(discourse_data, cookies)

    if not markdown_files:
        print("No markdown files created.")
        return

    print(f"Created {len(markdown_files)} markdown files")

    # Combine with existing markdowns
    print("Combining with existing markdowns...")
    combined_dir = combine_with_existing_markdowns(markdown_files)

    # Create index file
    create_index_file(combined_dir)

    print("\nProcess completed successfully!")
    print(
        f"Check the '{combined_dir}' folder for all combined markdown files.")
    print("Images are stored in the 'discourse_markdowns/images/' directory.")
    print("Check 'discourse_markdowns/images/image_index.md' for image descriptions.")


if __name__ == "__main__":
    main()
