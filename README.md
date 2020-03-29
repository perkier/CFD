Code used in CFD simulations for mesh creation and CFD results analysis

ANSYS Version used: ANSYS Fluent 18.0

# Table of contents

<!--ts-->
   * [Wings Aerodynamics](#wings-aerodynamics)
   * [Result Analysis](#result-analysis---mesh-independance)

<!--te-->


---


# Wings Aerodynamics

Python Package to:
- Automate the mesh creation process in ICEM CFD;
- Prepare journal files automaticly to be ran in ANSYS FLUENT;
- Data visualization of the results (Lift and Drag);
- Simulate flight times and distances;

## Applications

YYYYY

## Getting Started

YYYYY

## Main Components
YYYYYY

### Mesh_Creation
ABC ABc
- YYYYY
- YYYYY

### Journal_Creator
ABC ABc
- YYYYY
- YYYYY

### Journal_Creator
ABC ABc
- YYYYY
- YYYYY

### Post_Processing
ABC ABc
- YYYYY
- YYYYY

### Flight_Simulator
ABC ABc
- YYYYY
- YYYYY


---



# Result Analysis - Mesh Independance

The grid independence needs to be studied in order to minimize the impact of the grid size on discretization errors and computational cost. The grid size needs to be optimized to be small enough to guarantee that the results are independent of the mesh. The usual process is to increase 20-25% the mesh cells and compare the final flow results, defining an error-margin (normally 1-2%). However, there is no way to know where the difference is.
[Mesh Independance script](https://github.com/perkier/CFD/blob/master/Result_Analysis/Mesh_Independance_Analysis.py) compares a user-defined parameter (e.g. velocity) between the denser mesh and the original mesh. Finite Volumes techniques were used to approximate the denser mesh with the original one, as the coordinates of each cell varied between the meshes.


## Applications

![Gif of the Results](https://i.imgur.com/3T3feNc.gif)

## Getting Started
YYYYYY

## Main Components
- YYYYY
- YYYYY
- YYYYY
- YYYYY
- YYYYY



My Linkedin: https://www.linkedin.com/in/diogoncsa/
