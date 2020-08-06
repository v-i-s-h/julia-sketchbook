#=
# Sample program for ArgParse
=#

using ArgParse

function main()
    @show ARGS

    parser = ArgParseSettings()
    @add_arg_table parser begin
        "--opt1"
            help = "an option with an argument"
        "--opt2"
            help = "another option with an argument"
            arg_type = Int
            default = 0
        "--flag1"
            help = "an option without argument"
            action = :store_true
        "arg1"
            help = "a positional argument"
            required = true
    end

    parsed_args = parse_args(parser)
    println("Parsed args:")
    for (arg, val) in parsed_args
        println("$arg => $val")
    end
end

main()
