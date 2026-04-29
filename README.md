HERE IS THE LINK TO THE DATASET  https://www.kaggle.com/datasets/trungit/coco25k?resource=download

Mb we can try this article https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8/

AT THE MOMENT TWO VERSIONS GAN and RGB



## 🛠️ Environment & Installation Setup

This project can run on any computer (CPU, NVIDIA, AMD, or Mac). To avoid breaking other Python projects on your computer, it is highly recommended to use a Virtual Environment (`venv`).

**Step 1: Create a Virtual Environment**
Open your terminal, navigate to the project folder, and run this command to create a hidden environment folder named `.venv`:
`python -m venv .venv`
*(Note: If `python` doesn't work on Mac/Linux, try `python3 -m venv .venv`)*

**Step 2: Activate the Virtual Environment**
You must activate the environment before installing anything. Choose the command for your operating system:

* **Windows (Command Prompt):**
  `.venv\Scripts\activate.bat`
* **Windows (PowerShell):**
  `.venv\Scripts\Activate.ps1`
* **macOS / Linux:**
  `source .venv/bin/activate`

*(Success Check: You should now see `(.venv)` at the beginning of your terminal prompt line.)*

**Step 3: Install Universal Dependencies**
Now that you are inside the isolated environment, install the core libraries:
`pip install -r requirements.txt`

**Step 4: Install PyTorch for your specific hardware**
Choose the ONE command below that matches your computer:

* **Option A: I don't have a GPU (or I just want the easiest setup)**
  *This will run on any computer's CPU. It is slower, but guaranteed to work everywhere.*
  `pip install torch torchvision`

* **Option B: I have an NVIDIA GPU (Windows/Linux)**
  *This uses CUDA for massive speed boosts.*
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

* **Option C: I have a Mac (Apple Silicon M1/M2/M3)**
  *This automatically uses Apple's MPS acceleration.*
  `pip install torch torchvision`

* **Option D: I have an AMD GPU on Windows**
  *This uses Microsoft DirectML to bridge PyTorch to AMD cards.*
  `pip install torch torchvision torch-directml`
