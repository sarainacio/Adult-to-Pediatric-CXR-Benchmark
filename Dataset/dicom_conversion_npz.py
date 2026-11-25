from pathlib import Path
import time

import numpy as np
import pydicom
from tqdm import tqdm


# Compatibility with different pydicom versions.
try:
    from pydicom.pixels import apply_modality_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut


# ============================================================
# PATHS
# ============================================================

DICOM_DIR = Path("/path/to/vindr-cxr/dicom")

NPZ_DIR = Path("/path/to/vindr-cxr/npz")


# ============================================================
# CONFIGURATION
# ============================================================

# Skip files that have already been converted.
SKIP_EXISTING = True

# Supported file extensions.
DICOM_EXTENSIONS = {
    ".dcm",
    ".dicom",
}

# When the Modality LUT produces float64 values,
# convert them to float32 to reduce storage requirements.
#
# Use np.float64 to preserve the precision produced by pydicom.
FLOAT_DTYPE = np.float32


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_size(number_of_bytes):
    """
    Format a file size in bytes, KiB, MiB, and GiB.
    """

    kib = number_of_bytes / 1024
    mib = number_of_bytes / (1024 ** 2)
    gib = number_of_bytes / (1024 ** 3)

    return (
        f"{number_of_bytes:,} bytes | "
        f"{kib:,.2f} KiB | "
        f"{mib:,.2f} MiB | "
        f"{gib:,.3f} GiB"
    )


def invert_monochrome1(image):
    """
    Convert MONOCHROME1 intensity polarity to an
    equivalent MONOCHROME2 representation.

    Preserve the array value range and dtype.
    """

    original_dtype = image.dtype

    image_minimum = float(image.min())
    image_maximum = float(image.max())

    # Use float64 for the calculation to prevent overflow
    # with uint16 or int16 arrays.
    inverted = (
        image_maximum
        + image_minimum
        - image.astype(np.float64)
    )

    return inverted.astype(
        original_dtype,
        copy=False,
    )


# ============================================================
# DICOM READING AND INTERPRETATION
# ============================================================

def read_dicom_image(dicom_path):
    """
    Read a DICOM file and return an image that can be used
    without consulting the DICOM metadata again.

    Applies:
        - Pixel Data decoding;
        - Modality LUT or RescaleSlope/RescaleIntercept;
        - MONOCHROME1 polarity correction.

    Does not apply:
        - VOI LUT;
        - Window Center/Width;
        - normalization;
        - resize;
        - crop;
        - data augmentation.

    The array is returned at its original resolution.
    """

    dicom = pydicom.dcmread(
        str(dicom_path)
    )

    # --------------------------------------------------------
    # Decode Pixel Data
    # --------------------------------------------------------

    image = np.asarray(
        dicom.pixel_array
    )

    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D chest X-ray, "
            f"but received shape {image.shape}"
        )

    original_dtype = image.dtype

    # --------------------------------------------------------
    # Modality LUT / RescaleSlope / RescaleIntercept
    # --------------------------------------------------------

    image = apply_modality_lut(
        image,
        dicom,
    )

    modality_dtype = image.dtype

    # apply_modality_lut may return float64.
    # To reduce file size, store those values as float32.
    #
    # Arrays that remain uint16/int16 are preserved.
    if np.issubdtype(
        image.dtype,
        np.floating,
    ):
        image = image.astype(
            FLOAT_DTYPE,
            copy=False,
        )

    # --------------------------------------------------------
    # Validate pixel values
    # --------------------------------------------------------

    if not np.isfinite(image).all():
        raise ValueError(
            "The image contains NaN or Inf after "
            "applying the Modality LUT."
        )

    # --------------------------------------------------------
    # Correct MONOCHROME1 polarity
    # --------------------------------------------------------

    photometric_interpretation = str(
        getattr(
            dicom,
            "PhotometricInterpretation",
            "",
        )
    ).upper()

    was_inverted = False

    if photometric_interpretation == "MONOCHROME1":
        image = invert_monochrome1(
            image
        )

        was_inverted = True

    image = np.ascontiguousarray(
        image
    )

    information = {
        "original_dtype": str(original_dtype),
        "modality_dtype": str(modality_dtype),
        "saved_dtype": str(image.dtype),
        "shape": image.shape,
        "minimum": float(image.min()),
        "maximum": float(image.max()),
        "photometric_interpretation":
            photometric_interpretation,
        "monochrome1_inverted": was_inverted,
    }

    return image, information


# ============================================================
# ATOMIC NPZ SAVING
# ============================================================

def save_npz_atomic(output_path, image):
    """
    Save the image in a compressed NPZ archive.

    Write a temporary file first, so interrupted conversions
    do not leave incomplete final output files.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open("wb") as file:
            np.savez_compressed(
                file,
                image=image,
            )

        temporary_path.replace(
            output_path
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


# ============================================================
# FIND DICOM FILES
# ============================================================

def find_dicom_files(root_directory):
    """
    Recursively find DICOM files.
    """

    return sorted(
        path
        for path in root_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in DICOM_EXTENSIONS
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not DICOM_DIR.is_dir():
        raise FileNotFoundError(
            f"The DICOM directory does not exist:\n{DICOM_DIR}"
        )

    NPZ_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dicom_paths = find_dicom_files(
        DICOM_DIR
    )

    if not dicom_paths:
        raise FileNotFoundError(
            "No DICOM files were found in:\n"
            f"{DICOM_DIR}"
        )

    print()
    print(
        f"Number of DICOM files: "
        f"{len(dicom_paths):,}"
    )

    print(
        f"\nSource:\n{DICOM_DIR}"
    )

    print(
        f"\nDestination:\n{NPZ_DIR}"
    )

    start_time = time.perf_counter()

    processed_count = 0
    skipped_count = 0
    error_count = 0

    total_dicom_bytes = 0
    total_npz_bytes = 0

    progress_bar = tqdm(
        dicom_paths,
        desc="DICOM → NPZ",
    )

    for dicom_path in progress_bar:
        relative_path = dicom_path.relative_to(
            DICOM_DIR
        )

        output_path = (
            NPZ_DIR
            / relative_path
        ).with_suffix(".npz")

        if SKIP_EXISTING and output_path.exists():
            skipped_count += 1

            total_dicom_bytes += (
                dicom_path.stat().st_size
            )

            total_npz_bytes += (
                output_path.stat().st_size
            )

            progress_bar.set_postfix(
                processed=processed_count,
                skipped=skipped_count,
                errors=error_count,
            )

            continue

        try:
            image, information = read_dicom_image(
                dicom_path
            )

            save_npz_atomic(
                output_path,
                image,
            )

            dicom_size = dicom_path.stat().st_size
            npz_size = output_path.stat().st_size

            total_dicom_bytes += dicom_size
            total_npz_bytes += npz_size

            processed_count += 1

        except Exception as error:
            error_count += 1

            print(
                f"\nError processing:\n"
                f"{dicom_path}\n"
                f"{type(error).__name__}: {error}"
            )

        progress_bar.set_postfix(
            processed=processed_count,
            skipped=skipped_count,
            errors=error_count,
        )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    print("\nConversion finished.")

    print(
        f"\nProcessed: {processed_count:,}"
    )

    print(
        f"Already existing: {skipped_count:,}"
    )

    print(
        f"Errors: {error_count:,}"
    )

    print(
        "\nTotal size of the DICOM files considered:"
    )

    print(
        format_size(total_dicom_bytes)
    )

    print(
        "\nTotal size of the NPZ files:"
    )

    print(
        format_size(total_npz_bytes)
    )

    if total_dicom_bytes > 0:
        percentage = (
            total_npz_bytes
            / total_dicom_bytes
            * 100
        )

        reduction = 100 - percentage

        print(
            f"\nNPZ files represent "
            f"{percentage:.2f}% of the DICOM size."
        )

        print(
            f"Reduction: {reduction:.2f}%"
        )

    print(
        f"\nTotal time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    print(
        "\nFiles saved to:\n"
        f"{NPZ_DIR}"
    )


if __name__ == "__main__":
    main()
