# Basic AI Model
- A basic AI model (Still under development) with visualizations (Also under development) that will be used for digit recognition (28 by 28 pixels) at the end

## Compilation
- The project only has the source code but a few things are needed in order to compile.

### Windows
- Add [GLFW](https://www.glfw.org) **include** and **lib** folder to the complilation and **link** the library. i.e `g++ -I C:/glfw/path/include -L C:/glfw/path/lib -lglfw3` if using `g++`. I am using **GLFW v3.4** and the **mingw-w64** lib files, which means my `C:/glfw/path/lib` is actually `C:/glfw/path/lib-mingw-w64`
- GLFW in this project requires OpenGL, so make sure to link to that too, on Windows, the dll is available in system32 I think, so add `-lopengl32` and it should work fine since system32 in in **PATH**.
- GDI is also required in order to communicate with the GPU (I think), so link to it i.e `-lgdi32`
- Also define `_DEBUG` for some of the debug code to execute i.e `-D_DEBUG`
- I am using `g++` and VSCode task for my compilation and it goes something like this:
  ```
  "tasks": [
        {
            "label": "Build Debug",
            "type": "shell",
            "command": "g++",
            "args": [
                "-g",
                "-D_DEBUG",
                "-o",
                "bin/Debug/${workspaceFolderBasename}",
                "${workspaceFolder}\\src\\glad\\glad.c",
                "${workspaceFolder}\\src\\imgui\\imgui_draw.cpp",
                "${workspaceFolder}\\src\\imgui\\imgui_tables.cpp",
                "${workspaceFolder}\\src\\imgui\\imgui_widgets.cpp",
                "${workspaceFolder}\\src\\imgui\\imgui_impl_opengl3.cpp",
                "${workspaceFolder}\\src\\imgui\\imgui_impl_glfw.cpp",
                "${workspaceFolder}\\src\\imgui\\imgui.cpp",
                "${workspaceFolder}\\src\\main.cpp",
                "-I", "C:\\glfw\\path\\include",  // Path to glfw include file
                "-L", "C:\\glfw\\path\\lib-mingw-w64",  // Path to glfw lib file for mingw-w64
                "-lglfw3", "-lopengl32", "-lgdi32"
            ],
            "group": "build",
            "problemMatcher": [
                "$gcc"
            ],
            "dependsOn": "Copy Shaders (Debug)"
        },
        {
            "label": "Copy Shaders (Debug)", 
            "type": "shell",
            "command": "xcopy",
            "args": [
                "${workspaceFolder}\\src\\shaders",
                "${workspaceFolder}\\bin\\Debug\\shaders",
                "/i", "/e", "/y"
            ],
            "problemMatcher": []
        }
  ]
  ```
- You might have noticed that one of my tasks copies shader source files to **./shaders/** relative to the folder where the executable is. So make sure to automate that or copy the shaders to the exucutable's folder before running the executable after compilation. Otherwise the visualizations will not work.
- I have also introduce some training data to help get the neural network up before I try the digit recognition. The data is in the data folder and also has to be copied to **./data/**.
  ```
    // VS Code task to do the copying
    {
            "label": "Copy Data (Debug)",
            "type": "shell",
            "command": "xcopy",
            "args": [
                "${workspaceFolder}\\data",
                "${workspaceFolder}\\bin\\Debug\\data",
                "/i", "/e", "/y"
            ],
            "problemMatcher": []
        }
  ```
- The build task now just has to be tweaked a bit, replace `"dependsOn": "Copy Shaders (Debug)"` with `"dependsOn": ["Copy Shaders (Debug)", "Copy Data (Debug)"]`.
- Note: ~~I plan on using compute shaders for training in the near future, so I will hardly do the CPU version... I apologize to those who do not have dedicated GPUs in advance, but since this is public, surely someone will volunteer to handle that part.~~ I have implemented and use a compute shader for the forwarding and it seems to work fine even with an integraded GPU while simultaniously showing the incomplete visualizations.

### Unix
- I have no idea, tough luck
