# Full Context Dialogue Enhancement for Video Analysis Pipeline

## Problem

The video analysis pipeline was experiencing rate limit errors during dialogue enhancement because it was processing scenes in batches. This batching approach was fragmenting context and preventing effective character identification across scenes.

## Solution

We've implemented a new dialogue enhancement approach that:

1. **Processes the entire structured analysis JSON in a single API call** to maintain full context
2. **Returns only the dialogue label mappings** (not the full JSON) to minimize response size
3. **Applies these mappings back to the original JSON** structure
4. **Handles rate limits robustly** with exponential backoff and intelligent retry logic

## Implementation Details

### 1. Core Components

- **`full_context_dialogue_enhancer.py`**: Standalone module that implements the full context approach
  - `enhance_dialogue_with_full_context()`: Main function that processes the entire JSON
  - `create_prompt_data()`: Creates a simplified version of the data for the prompt
  - `get_dialogue_labels_with_retries()`: Makes API calls with robust retry logic
  - `apply_dialogue_labels()`: Applies the label mappings back to the original JSON

### 2. Pipeline Integration

- **`utils/pipeline_enhancer.py`**: Updated to patch the pipeline's dialogue enhancement method
  - `enhance_dialogue_processor()`: Replaces the original method with our full context approach
  - Seamlessly integrates with the existing pipeline without changing core files

### 3. Key Improvements

- **Full Context Processing**: Analyzes all scenes at once to maintain context
- **Focused API Response**: Only returns the dialogue label mappings, not the entire JSON
- **Robust Rate Limit Handling**: 
  - Implements exponential backoff with jitter
  - Extracts rate limit information from error messages
  - Uses longer timeout for large context processing
- **Debugging Support**: Saves API responses and extracted JSON to help diagnose issues

## Usage

### Integrated with Pipeline

The enhancement is automatically applied when the pipeline runs, with no changes needed to your workflow:

```bash
python run_pipeline.py <video_path>
```

### Standalone Mode

You can also run the dialogue enhancement separately on an existing structured analysis:

```bash
python full_context_dialogue_enhancer.py --analysis <path_to_structured_analysis.json>
```

### Testing

A test script is provided to verify the enhancement works correctly:

```bash
python test_dialogue_enhancer.py --input <path_to_structured_analysis.json>
```

## Benefits

1. **Better Character Identification**: Maintains full context across scenes for more consistent character naming
2. **Reduced API Calls**: Makes a single API call instead of multiple batch calls
3. **Rate Limit Resilience**: Intelligently handles rate limits with appropriate retry logic
4. **Minimized Modification**: Integrates with the existing pipeline without changing core files

## Technical Notes

- Uses the Gemini 1.5 Pro model specifically for its larger context window
- Implements thread-based timeout handling to prevent hanging on API calls
- Includes detailed logging for better error diagnosis
- Stores debugging information in a `debug` directory for error analysis