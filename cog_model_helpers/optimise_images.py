from PIL import Image

IMAGE_FILE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
FORMAT_CHOICES = ["webp", "jpg", "png"]
DEFAULT_FORMAT = "webp"
DEFAULT_QUALITY = 95


def predict_output_format() -> str:
    return DEFAULT_FORMAT


def predict_output_quality() -> int:
    return DEFAULT_QUALITY


def should_optimise_images(output_format: str, output_quality: int):
    return output_quality < 100 or output_format in [
        "webp",
        "jpg",
    ]


def optimise_image_files(
    output_format: str = DEFAULT_FORMAT, output_quality: int = DEFAULT_QUALITY, files=[]
):
    if should_optimise_images(output_format, output_quality):
        optimised_files = []
        for file in files:
            if file.is_file() and file.suffix in IMAGE_FILE_EXTENSIONS:
                image = Image.open(file)
                optimised_file_path = file.with_suffix(f".{output_format}")
                image.save(
                    optimised_file_path,
                    quality=output_quality,
                    optimize=True,
                )
                optimised_files.append(optimised_file_path)
            else:
                optimised_files.append(file)

        return optimised_files
    else:
        return files
