#!/bin/bash

# Script to create submission zip file
# Usage: ./create_submission.sh [directory_name]

# Get the current directory name (should be yourname_ps2)
CURRENT_DIR=$(basename "$(pwd)")

# If a directory name is provided as argument, use it
if [ -n "$1" ]; then
    DIR_NAME="$1"
    SOURCE_DIR="$1"
# Check if we're in a directory ending with _ps2
elif [[ "$CURRENT_DIR" == *_ps2 ]]; then
    DIR_NAME="$CURRENT_DIR"
    SOURCE_DIR="."
# Otherwise, look for a subdirectory ending with _ps2
else
    # Find the first directory ending with _ps2
    FOUND_DIR=$(find . -maxdepth 1 -type d -name "*_ps2" | head -1)
    if [ -n "$FOUND_DIR" ]; then
        DIR_NAME=$(basename "$FOUND_DIR")
        SOURCE_DIR="$FOUND_DIR"
        echo "Found submission directory: $DIR_NAME"
    else
        echo "Error: No directory ending with '_ps2' found"
        echo "Please run this script from inside the submission directory or provide the directory name as an argument"
        exit 1
    fi
fi

# Create zip file name
ZIP_NAME="${DIR_NAME}.zip"

# Remove old zip if it exists
if [ -f "$ZIP_NAME" ]; then
    echo "Removing existing $ZIP_NAME"
    rm "$ZIP_NAME"
fi

# Create a temporary directory for the submission structure
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Create the submission directory
SUBMIT_DIR="$TEMP_DIR/$DIR_NAME"
mkdir -p "$SUBMIT_DIR"

echo "Creating $ZIP_NAME from $SOURCE_DIR..."

# Copy data.pickle
if [ -f "$SOURCE_DIR/data.pickle" ]; then
    cp "$SOURCE_DIR/data.pickle" "$SUBMIT_DIR/"
    echo "  Added data.pickle"
else
    echo "  Warning: data.pickle not found in $SOURCE_DIR"
fi

# Copy all Python files
PY_COUNT=0
for py_file in "$SOURCE_DIR"/*.py; do
    if [ -f "$py_file" ]; then
        cp "$py_file" "$SUBMIT_DIR/"
        echo "  Added $(basename "$py_file")"
        ((PY_COUNT++))
    fi
done

if [ $PY_COUNT -eq 0 ]; then
    echo "  Warning: No Python files found in $SOURCE_DIR"
fi

# Copy solve_4R.mp4
if [ -f "$SOURCE_DIR/solve_4R.mp4" ]; then
    cp "$SOURCE_DIR/solve_4R.mp4" "$SUBMIT_DIR/"
    echo "  Added solve_4R.mp4"
else
    echo "  Warning: solve_4R.mp4 not found in $SOURCE_DIR"
fi

# Create the zip file from the temporary directory
cd "$TEMP_DIR"
zip -r "$OLDPWD/$ZIP_NAME" "$DIR_NAME" > /dev/null
cd - > /dev/null

echo ""
echo "Successfully created $ZIP_NAME"
echo ""
echo "Zip file contents:"
unzip -l "$ZIP_NAME" | tail -n +4 | head -n -2

