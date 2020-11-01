#=
    Dash for Julia - Tutorial - Callbacks 1
    Source: https://dash-julia.plotly.com/basic-callbacks
=#

using  Dash, DashCoreComponents, DashHtmlComponents

app = dash()

app.layout = html_div() do
    dcc_input(id="input-3", value="some value", type="text"),
    html_div(id="output-1")
end

callback!(app, Output("output-1", "children"), Input("input-3", "value")) do input_value
    "You've entered $(input_value)"
end

run_server(app, "0.0.0.0", debug=true)