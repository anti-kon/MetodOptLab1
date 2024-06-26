var lexer, parseId;

tokens = ""

function parse() {
    if (parseId) {
        window.clearTimeout(parseId);
    }

    parseId = window.setTimeout(function () {
        var code, str,
            lexer, tokens, token, i,
            parser, syntax;

        code = document.getElementById('equation').value;
        if (code === ''){
            document.getElementById('equation_showcase').innerHTML = ''
            return
        }
        try {
            if (typeof lexer === 'undefined') {
                lexer = new TapDigit.Lexer();
            }

            if (typeof parser === 'undefined') {
                parser = new TapDigit.Parser();
            }

            tokens = [];
            lexer.reset(code);
            while (true) {
                token = lexer.next();
                if (typeof token === 'undefined') {
                    break;
                }
                tokens.push(token);
            }

            str = '<table width=200>\n';
            for (i = 0; i < tokens.length; i += 1) {
                token = tokens[i];
                str += '<tr>';
                str += '<td>';
                str += token.type;
                str += '</td>';
                str += '<td align=center>';
                str += token.value;
                str += '</td>';
                str += '</tr>';
                str += '\n';
            }
            str = tokens;

            syntax = parser.parse(code);

            function stringify(object, key, depth) {
                var indent = '',
                    str = '',
                    value = object[key],
                    i,
                    len;

                while (indent.length < depth * 3) {
                    indent += ' ';
                }

                switch (typeof value) {
                case 'string':
                    str = value;
                    break;
                case 'number':
                case 'boolean':
                case 'null':
                    str = String(value);
                    break;
                case 'object':
                    for (i in value) {
                        if (value.hasOwnProperty(i)) {
                            str += ('<br>' + stringify(value, i, depth + 1));
                        }
                    }
                    break;
                }

                return indent + ' ' + key + ': ' + str;
            }

            console.log(syntax)
            document.getElementById('syntax').innerHTML = stringify(syntax, 'Expression', 0);
            toMathMl(syntax.Expression).then(e => {
                convertStringToMath(e).then(s => document.getElementById('equation_showcase').innerHTML = s)
            });
        } catch (e) {
            document.getElementById('syntax').innerText = e.message;
        }
        parseId = undefined;
    }, 345);
}
