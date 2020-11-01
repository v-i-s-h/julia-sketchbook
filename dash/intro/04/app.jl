#=
    Dash for Julia - Tutorial 4 - Markdown
    Source: https://dash-julia.plotly.com/getting-started
=#

using Dash, DashCoreComponents, DashHtmlComponents

app = dash()
md_text = "
### Dash and Markdown

Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org)
specification of Markdown.
"

app.layout = html_div() do
    dcc_markdown(md_text)
end

run_server(app, "0.0.0.0", debug=true)