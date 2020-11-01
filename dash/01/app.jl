using Dash, DashHtmlComponents, DashCoreComponents

app = dash()

app.layout = html_div(style=Dict("backgroundColor"=>"#111111")) do
    html_h1(
        "Hello Dash!",
        style = Dict("color"=>"#7FDBFF", "textAlign"=>"center"),
    ),
    html_div(
        "Dash: A Web Application framework for Julia",
        style=Dict("color"=>"#7FDBFF")
    ),
    dcc_graph(
        id = "example-graph-1",
        figure = (
            data = [
                (
                    x = ["giraffes", "orangutans", "monkeys"], 
                    y = [20, 14, 23], 
                    type = "bar",
                    name = "SF"),
                (
                    x = [ "giraffes", "orangutans", "monkeys"],
                    y = [12, 18, 29],
                    type = "bar",
                    name = "Montreal"
                )
            ],
            layout = (
                title="Dash Data Visualization",
                barmode="group",
                plot_bgcolor="#000000",
                paper_bgcolor="#111111",
                font=Dict("color"=>"#7FDBFF")
            )
        )
    )
end

run_server(app, "0.0.0.0", debug=true)