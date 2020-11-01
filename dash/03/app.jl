#=
    Dash for Julia - Tutorial 3 - More About Visualization
    Source: https://dash-julia.plotly.com/getting-started
=#

using DataFrames, CSV, PlotlyJS, RDatasets
using Dash, DashCoreComponents, DashHtmlComponents

iris = dataset("datasets", "iris")

pl = Plot(
    iris,
    x = :SepalLength,
    y = :SepalWidth,
    mode="markers",
    marker_size=8,
    group=:Species
)

app = dash()

app.layout = html_div() do
    html_h4("Iris Sepal Length vs Sepal Width"),
    dcc_graph(
        id = "example-graph-3",
        figure = pl
    )
end

run_server(app, "0.0.0.0", debug=true)