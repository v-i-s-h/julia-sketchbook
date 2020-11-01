#=
    Dash for Julia - Turotial - Callbacks with multiple inputs
    Source: https://dash-julia.plotly.com/basic-callbacks
=#

using DataFrames, PlotlyJS, CSV
using Dash, DashCoreComponents, DashHtmlComponents

datafile = "./data/country_indicators.csv"
df = CSV.File(datafile) |> DataFrame
dropmissing!(df)
rename!(df, Dict(:"Year"=>"year"))

available_indicators = unique(df[!, "Indicator Name"])
years = unique(df[!, "year"])

app = dash()

app.layout = html_div() do
    html_div(
        children = [
            dcc_dropdown(
                id = "xaxis-column",
                options = [
                    (label=i, value=i) for i in available_indicators
                ],
                value = rand(available_indicators)
            ),
            dcc_radioitems(
                id = "xaxis-type",
                options = [
                    (label=i, value=i) for i in ["linear", "log"]
                ],
                value = "linear"
            ),
        ],
        style = (width="48%", display="inline-block"),
    ),
    html_div(
        children = [
            dcc_dropdown(
                id = "yaxis-column",
                options = [
                    (label=i, value=i) for i in available_indicators
                ],
                value = rand(available_indicators)
            ),
            dcc_radioitems(
                id = "yaxis-type",
                options = [
                    (label=i, value=i) for i in ["linear", "log"]
                ],
                value = "linear"
            )
        ],
        style = (width="48%", display="inline-block", float="right")
    ),
    dcc_graph(id="indicator-graphic", style = (width="48%", display="inline-block", float="center")),
    dcc_slider(
        id = "year-slider-2",
        min = minimum(years),
        max = maximum(years),
        marks = Dict([Symbol(v) => Symbol(v) for v in years]),
        value = minimum(years),
        step = nothing
    )
end

callback!(
    app,
    Output("indicator-graphic", "figure"),
    Input("xaxis-column", "value"),
    Input("yaxis-column", "value"),
    Input("xaxis-type", "value"),
    Input("yaxis-type", "value"),
    Input("year-slider-2", "value"),
) do xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type, year_value
    df2 = df[df.year .== year_value, :]
    return Plot(
        df2[df2[!, Symbol("Indicator Name")] .== xaxis_column_name, :Value],
        df2[df2[!, Symbol("Indicator Name")] .== yaxis_column_name, :Value],
        Layout(
            xaxis_type = xaxis_type == "linear" ? "linear" : "log",
            xaxis_title = xaxis_column_name,
            yaxis_type = yaxis_type == "linear" ? "linear" : "log",
            yaxis_title = yaxis_column_name,
            hovermode = "closest"
        ),
        kind = "scatter",
        text = df2[
            df2[Symbol("Indicator Name")] .== yaxis_column_name,
            Symbol("Country Name")
        ],
        mode = "markers",
        marker_size = 15,
        marker_opacity = 0.50,
        marker_line_width = 0.50,
        marker_line_color = "white"
    )
end

run_server(app, "0.0.0.0", debug=true)
