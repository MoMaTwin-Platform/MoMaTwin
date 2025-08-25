#!/bin/bash

# count_minmax.sh - Run count_minmax.py on multiple directories

# Set output directory
OUTPUT_DIR="spellwords_action_stats"

# List of directories to analyze (edit these paths as needed)
DIRS=(
  "/x2robot/zhengwei/10009/20250416-day-spell-word"
  "/x2robot/zhengwei/10009/20250415-day-spell-word"
  "/x2robot/zhengwei/10009/20250414-day-spell-word"
  "/x2robot/zhengwei/10009/20250411-day-spell-word"
  "/x2robot/zhengwei/10009/20250410-day-spell-word"
  "/x2robot/zhengwei/10009/20250407-day-spell-word"
  "/x2robot/zhengwei/10009/20250403-day-spell-word"
  "/x2robot/zhengwei/10009/20250402-night-spell-word"
  "/x2robot/zhengwei/10009/20250402-day-spell-word"
  "/x2robot/zhengwei/10009/20250401-day-spell-word"
  "/x2robot/zhengwei/10009/20250331-day-spell-word-car-2"
  "/x2robot/zhengwei/10009/20250331-day-spell-word-car"
  # "/x2robot/zhengwei/10078/20250427-day-pick_up-tissue"
  # "/x2robot/zhengwei/10078/20250425-day-pick_up-tissue"
  # Add more directories as needed
)

# Print summary
echo "Will analyze ${#DIRS[@]} directories:"
for dir in "${DIRS[@]}"; do
  echo "  - $dir"
done

# Validate directories
VALID_DIRS=()
for dir in "${DIRS[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "Warning: Directory does not exist: $dir"
  elif [ ! -f "$dir/report.json" ]; then
    echo "Warning: No report.json found in: $dir"
  else
    VALID_DIRS+=("$dir")
  fi
done

# Check if we have any valid directories
if [ ${#VALID_DIRS[@]} -eq 0 ]; then
  echo "Error: No valid directories to process."
  exit 1
fi

# Run the Python script
echo "Starting analysis..."
python3 count_minmax.py "$OUTPUT_DIR" "${VALID_DIRS[@]}"

# Check if the analysis was successful
if [ $? -eq 0 ]; then
  echo "Analysis completed successfully!"
  echo "Results saved to: $OUTPUT_DIR"
else
  echo "Analysis failed."
fi