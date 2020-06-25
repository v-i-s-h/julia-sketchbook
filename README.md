# Julia Sketchbook

This repository is a collection of random julia scripts I collected from internet. 
For most of the projects, I have tried to include the original source of the content
at the top header.

## How to use?
Each folder itself is a self contained project with `Project.toml` and `Manifest.toml`.
You can run the following in each directory to create a reproducible environment
```
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

## List of Projects
| Project | Description |
|:--------| ----------- |
| [optimizing-kalman](./optimizing-kalman/) | An excercise on optimizing julia code. |
| [mle](./mle/) | Maximum Likelihood Estimation for Logistic regression. |