var lexer, parseId;

tokens = ""

async function parse(idEquation, showcase, inequality, inequalitySignId, inequalityRightId) {
    // if (parseId) {
    //     window.clearTimeout(parseId);
    // }

    parseId = window.setTimeout(function () {
        var code, str,
            lexer, tokens, token, i,
            parser, syntax;

        code = document.getElementById(idEquation).value;
        if (code.length === 0){
            document.getElementById(showcase).innerHTML = `<mo>...</mo>`;
            if (inequality) {
                if (document.getElementById(inequalitySignId).value === '≤')
                    document.getElementById(showcase).innerHTML += `<mo>≤</mo>`;
                else if (document.getElementById(inequalitySignId).value === '≥')
                    document.getElementById(showcase).innerHTML += `<mo>≥</mo>`;
                else
                    document.getElementById(showcase).innerHTML += `<mo>=</mo>`;
                document.getElementById(showcase).innerHTML +=
                    document.getElementById(inequalityRightId).value === '' ? `<mn>0</mn>` :
                    `<mn>${document.getElementById(inequalityRightId).value}</mn>`;
            }
            return
        }

        if (inequality) {
            if (document.getElementById(inequalitySignId).value === '≤') {
                code = 'min(' + code;
            } else if (document.getElementById(inequalitySignId).value === '≥') {
                code = 'max(' + code;
            }else {
                code = 'equals(' + code;
            }
            code += ', '
            code += document.getElementById(inequalityRightId).value === '' ? 0 :
                document.getElementById(inequalityRightId).value;
            code += ')';
        }

        console.log(code);

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
            toMathMl(syntax.Expression).then(e => {
                convertStringToMath(e).then(s => {document.getElementById(showcase).innerHTML = s;})
            });
        } catch (e) {
            console.log(e)
        }
        parseId = undefined;
    }, 345);
}
