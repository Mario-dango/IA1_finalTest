# Copilot Instructions for IA1_finalTest

## Project Overview
This project is the final assignment for the "Inteligencia Artificial 1" course (2022). It is a Python application organized in a classic MVC (Model-View-Controller) structure, focused on image-based AI tasks (likely classification or recognition of mechanical parts).

## Architecture & Key Components
- **app/main.py**: Entry point. Initializes the application and connects MVC components.
- **app/controller/**: Contains controllers (e.g., `main_controller.py`) that mediate between the view and model.
- **app/model/**: Core logic and data models, including:
  - `image_model.py`: Image processing/feature extraction
  - `prediction_model.py`: Prediction/classification logic
  - `astar_model.py`, `strips_model.py`: AI search/planning algorithms
  - `database_model.py`: Data persistence (if used)
- **app/view/**: UI components (likely PyQt or similar), e.g., `main_window.py`, `image_view.py`, `grid_astar.py`.
- **app/images/**: Contains datasets and sample images, organized by type (arandelas, clavos, tornillos, tuercas).

## Developer Workflows
- **Run the app**: Execute `python app/main.py` from the project root.
- **No explicit build step**: Pure Python, no compilation required.
- **Testing**: No standard test suite detected; manual testing via the UI is likely.
- **Debugging**: Use print statements or IDE debugger. Key files: `main.py`, `main_controller.py`, `main_window.py`.

## Project-Specific Conventions
- **MVC separation**: Keep logic in `model/`, UI in `view/`, and glue code in `controller/`.
- **Image datasets**: Use the provided folder structure for training/testing images. Do not hardcode paths; use relative paths from `app/images/`.
- **Class naming**: Models and controllers are named after their function (e.g., `AStarModel`, `MainController`).
- **No external config**: All configuration is in code; no `.env` or config files detected.

## Integration & Dependencies
- **Likely uses OpenCV, NumPy, PyQt**: Check imports in `model/` and `view/` for required packages.
- **No requirements.txt**: If adding dependencies, create one in the project root.

## Examples
- To add a new image processing algorithm, create a new file in `model/` and connect it via the controller.
- To add a new UI feature, update the relevant `view/` file and ensure the controller mediates any model interaction.

## References
- See `README.md` for a brief project description.
- Key files: `app/main.py`, `app/controller/main_controller.py`, `app/model/image_model.py`, `app/view/main_window.py`.

---
For questions about project structure or conventions, follow the patterns in the existing codebase. If unsure, ask for clarification or propose changes in a modular way.
