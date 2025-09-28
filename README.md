# AI for Fitness Project

This project is designed to use AI for fitness applications, including tracking and analyzing exercises such as push-ups and lunges through media files and Python scripts.

## Prerequisites

To run this project, you need to have the following installed:

- Python 3.9 (recommended version)
- pip (Python package manager)
- VS Code

## Setting Up the Project

### 1. **Clone the Repository**

To get started, clone the repository to your local machine using the following command in the command prompt:

```bash
git clone https://github.com/fantasticfahim/ai_for_fitness.git
````

### 2. **Create a Virtual Environment**

It's a good practice to use a virtual environment to manage dependencies. Navigate to the project folder and create a virtual environment:

```bash
cd ai_for_fitness
python3.9 -m venv myenv
```

### 3. **Activate the Virtual Environment**

Activate the virtual environment you just created:

* **Windows (Command Prompt):**

  ```bash
  myenv\Scripts\activate
  ```

* **Windows (PowerShell):**

  ```bash
  .\myenv\Scripts\Activate.ps1
  ```

* **Linux/macOS:**

  ```bash
  source myenv/bin/activate
  ```

### 4. **Install Dependencies**

With the virtual environment activated, install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install all the necessary libraries such as `mediapipe`, `numpy`, and others for the project to work properly.

### 5. **Run the Project**

* **Running the .ipynb Jupyter Notebook

Open the Media Pipe Pose Demo Tutorial.ipynb file in VS Code.

Click on the Run button for each cell, or run the entire notebook by using the Run All option in the toolbar at the top.

* **Running the .py Python Script

Open any .py file (e.g., pushupWork.py or LegLungeWork.py) in VS Code.

Make sure the virtual environment is activated by opening command pallete and selecting the python 3.9 version.

You can run the .py script by opening the Terminal in VS Code (press Ctrl+` ) and running the following command:

```bash
python pushupWork.py
```

### 6. **Assets**

The `Assets` folder contains video files and Python scripts for different exercises like **push-ups** and **lunges**. You can use these assets to track and analyze different fitness exercises.

* **LegLunge.mp4**: Video for lunges exercise
* **pushup.mp4**: Video for push-up exercise
* **LegLungeWork.py**: Python script for lunges analysis
* **pushupWork.py**: Python script for push-up analysis

### 7. **Notes**

* Make sure to use Python 3.9 to avoid compatibility issues.
* The `myenv` folder should not be pushed to the repository, as it contains local dependencies. Please recreate the virtual environment when setting up the project.
* You can modify the video files or Python scripts in the **Assets** folder to track other exercises or add more analysis features.

## License

This project is open-source and available under the [MIT License](LICENSE).
This template includes everything from cloning the repository to setting up the environment, running the project, and managing the assets.
