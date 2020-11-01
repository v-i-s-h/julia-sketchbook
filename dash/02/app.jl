# #=
#     Dash for Julia - Tutorial 2 - Resuable Components
#     Source: https://dash-julia.plotly.com/getting-started
# =#

using DataFrames, UrlDownload, Dash, DashCoreComponents, DashHtmlComponents

# # Get data file
datafile = "https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv"
df3 = urldownload(datafile, true) |> DataFrame

function generate_table(df, max_rows=10)
    html_table([
        html_thead(html_tr([html_th(col) for col in names(df)])),
        html_tbody([
            html_tr([html_td(df[r, c]) for c in names(df)]) for r in 1:min(nrow(df), max_rows)
        ])
    ])
end

app = dash()

app.layout = html_div() do
    html_h4("US Ag Exp (2011)"),
    generate_table(df3, 10)
end

run_server(app, "0.0.0.0", debug=true)