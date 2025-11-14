from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string
import io
import os


class SimpleCaptcha:
    def __init__(self, width=160, height=60, fonts=None, font_size=42):
        """
        Initialize the SimpleCaptcha generator.
        - width, height: dimensions of the captcha image.
        - fonts: list of paths to .ttf fonts to use.
        - font_size: default font size for characters.
        """
        self.width = width
        self.height = height
        self.font_size = font_size
        # Default to a common system font if none is provided (FOR MACOS)
        self.fonts = fonts or ["/System/Library/Fonts/Supplemental/Arial.ttf"]
        self._load_fonts()

    def _load_fonts(self):
        """Load all valid font files into memory."""
        self.truefonts = []
        for font_path in self.fonts:
            if os.path.exists(font_path):
                try:
                    # Load font with specified size
                    self.truefonts.append(ImageFont.truetype(font_path, self.font_size))
                except Exception:
                    pass
        if not self.truefonts:
            raise RuntimeError("No valid fonts found!")

    def _random_color(self, start=0, end=255):
        """Return a random RGB color between given ranges."""
        return (random.randint(start, end),
                random.randint(start, end),
                random.randint(start, end))

    def _add_noise(self, draw):
        """Add random noise (dots and lines) to make CAPTCHA harder to read by bots."""
        # Add random colored dots
        for _ in range(random.randint(100, 150)):
            x, y = random.randint(0, self.width), random.randint(0, self.height)
            draw.point((x, y), fill=self._random_color(100, 255))

        # Add random lines
        for _ in range(random.randint(5, 10)):
            start = (random.randint(0, self.width), random.randint(0, self.height))
            end = (random.randint(0, self.width), random.randint(0, self.height))
            draw.line([start, end], fill=self._random_color(50, 150), width=1)

    def _draw_text(self, base_image, text):
        """
        Draw each character on the image with:
        - random rotation
        - random vertical offset
        - small random horizontal overlap
        """
        draw = ImageDraw.Draw(base_image)
        x_offset = 10  # start position for text

        for char in text:
            # Random font and color for each character
            font = random.choice(self.truefonts)
            color = self._random_color(10, 150)

            # Create a transparent image for the character (so rotation doesn’t mess up the base)
            char_image = Image.new('RGBA', (self.font_size * 2, self.height * 2), (255, 255, 255, 0))
            char_draw = ImageDraw.Draw(char_image)

            # Random vertical offset (character "bounces" up and down)
            y_offset = random.randint(0, self.height // 8)

            # Draw the character on its own layer
            char_draw.text((10, y_offset), char, font=font, fill=color)

            # Apply a random rotation between -20° and +20°
            angle = random.uniform(-15, 15)
            rotated_char = char_image.rotate(angle, expand=1)

            # Compute character bounding box to determine how much to move next one
            bbox = rotated_char.getbbox()
            if bbox:
                char_width = bbox[2] - bbox[0]
            else:
                char_width = self.font_size

            # Paste the rotated character onto the main image (RGBA support)
            base_image.paste(rotated_char, (int(x_offset), 0), rotated_char)

            # Random overlap (negative offset means overlap)
            x_offset += char_width + 5 - random.randint(0, 8)

        return base_image

    def generate_image(self, text=None):
        """
        Generate a CAPTCHA image.
        - If text not given, generate random 5-character alphanumeric string.
        Returns: (PIL Image, text)
        """
        # Generate random text if none provided
        if text is None:
            text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(3, 4)))

        # Create blank background image with random light color
        bg_color = self._random_color(200, 255)
        image = Image.new('RGB', (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(image)

        # Add noise before text (so characters appear over it)
        self._add_noise(draw)

        # Draw characters with rotation and offset
        image = self._draw_text(image, text)

        # Add a slight smooth filter to make it less perfect
        image = image.filter(ImageFilter.SMOOTH)

        return image, text

    def generate(self, text=None, format="PNG"):
        """Generate the CAPTCHA and return as bytes buffer (like ImageCaptcha.generate)."""
        image, text = self.generate_image(text)
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return buffer, text

    def write(self, text, output_path):
        """Generate and save CAPTCHA image to a file."""
        image, _ = self.generate_image(text)
        image.save(output_path, format="PNG")


# Example usage
if __name__ == "__main__":
    captcha = SimpleCaptcha(fonts=['fonts/FontdinerSwanky-Regular.ttf'])
    for i in range(3):
        image, text = captcha.generate_image()
        print("Generated CAPTCHA text:", text)
        image.show()
