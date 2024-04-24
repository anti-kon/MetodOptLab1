async function getZeutendijkMethodPossibleDirectionsResult() {
	let variablesNum = parseInt(document.getElementById('variables_num').value);
    let restrictionsNum = parseInt(document.getElementById('restrictions_num').value);
	document.getElementById("loader").style.display='flex';
	let save_formula = document.getElementById('equation_showcase').cloneNode(true);
	save_formula.id = '';
	save_formula.className = '';
    let functionEquation = document.getElementById('functionEquation').value;
	const restrictions = new Array(restrictionsNum);
	for (let restrictionIndex = 0; restrictionIndex < restrictionsNum; restrictionIndex++) {
		restrictions[restrictionIndex] = {
			equation: document.getElementById(`matrix${restrictionIndex}Equation`).value,
			sign: document.getElementById(`row${restrictionIndex}Select`).value,
			value: document.getElementById(`row${restrictionIndex}RightInput`).value === '' ? 0 :
                   document.getElementById(`row${restrictionIndex}RightInput`).value
		};
	}
	// const signRestrictions = new Array(variablesNum);
	// for (let signRestrictionIndex = 0; signRestrictionIndex < variablesNum; signRestrictionIndex++) {
	// 	signRestrictions[signRestrictionIndex] = {
	// 		equation: document.getElementById(`x${signRestrictionIndex + 1}Name`).value,
	// 		sign: document.getElementById(`variable${signRestrictionIndex}Select`).value,
	// 		value: document.getElementById(`variable${signRestrictionIndex}SignValue`).value === '' ? 0 :
    //                document.getElementById(`variable${signRestrictionIndex}SignValue`).value
	// 	};
	// }
	let lambdaValue = document.getElementById('lambda').value === '' ? (1) :
		parseFloat(document.getElementById('lambda').value);
	let epsilon = document.getElementById('epsilon').value === '' ? (0.001) :
		parseFloat(document.getElementById('epsilon').value);

	if (functionEquation === '') {
		document.getElementById("loader").style.display='none';
        return;
    }
	restrictions.forEach((element) => {
		if (element.equation === '') {
			document.getElementById("loader").style.display='none';
			return;
		}
	});

    let output = document.getElementById("output");
    let result = document.createElement('div');
    result.className = 'result';
	let calculations;
	try {
		calculations = await eel.get_zeutendijk_method_possible_directions_result(
			functionEquation, restrictions, /*signRestrictions*/[], lambdaValue, epsilon, variablesNum
		)();
		document.getElementById("loader").style.display='none';
	} catch (e) {
		document.getElementById("loader").style.display='none';
		result.innerHTML += "Ошибка в входных данных";
		output.prepend(result);
		return
	}

	console.log(calculations)
	result.innerHTML += "Исходная функция:"
	result.appendChild(document.createElement('br'))
	result.appendChild(save_formula)
	result.innerHTML += `Коэффициент дробления шага: ${lambdaValue}`
	result.appendChild(document.createElement('br'))
	result.innerHTML += `Точность: ${epsilon}`
	result.appendChild(document.createElement('br'))
	result.innerHTML += `Количество итераций: ${calculations[1]}`
	result.appendChild(document.createElement('br'));
	result.innerHTML += `Путь:`;
	result.appendChild(document.createElement('br'));
	let path = document.createElement('div');
	path.style.display = 'grid';
	path.style.gridTemplateColumns = `repeat(${variablesNum + 1}, max-content)`;
	path.appendChild(document.createElement('div'));
	path.className = 'pathMatrix';
	path.lastChild.innerHTML = 'Итерация';
	for (let i = 0; i < variablesNum; i++) {
		path.appendChild(document.createElement('div'));
		path.lastChild.innerHTML = `<math><msub><mi>x</mi><mn>${i + 1}</mn></msub></math>`;
	}
	for (let i = 0; i < calculations[0].length; i++) {
		path.appendChild(document.createElement('div'));
		path.lastChild.innerHTML = parseInt(calculations[0][i][0]);
		for (let j = 1; j < calculations[0][i].length; j++) {
			path.appendChild(document.createElement('div'));
			path.lastChild.innerHTML = parseFloat(calculations[0][i][j]).toFixed(3);
		}
	}
	result.appendChild(path)
	result.appendChild(document.createElement('br'))
	result.innerHTML += `Ответ: (`;
	result.innerHTML += parseFloat(calculations[2][1]).toFixed(3);
	for (let i = 2; i < calculations[2].length; i++){
		result.innerHTML += ', ';
		result.innerHTML += parseFloat(calculations[2][i]).toFixed(3);
	}
	result.innerHTML += ') '
	result.innerHTML += parseFloat(calculations[3]).toFixed(3);
	result.appendChild(document.createElement('br'))
	let img1 = new Image();
	img1.onload = function(){
	  // execute drawImage statements here
	};
	img1.src = calculations[5];
	result.appendChild(img1);
	output.prepend(result);
}

async function handleFileLoad(event) {
    const file = document.getElementById('file_input').files[0];
    const reader = new FileReader();

    reader.addEventListener(
        "load",
        () => {
            let data = reader.result.match(/(-?\d+(\.\d+)?)|([<>=])/g);
            let variablesNum = parseInt(data[0]);
            let restrictionsNum = parseInt(data[1]);

            document.getElementById('variables_num').value = variablesNum;
            document.getElementById('restrictions_num').value = restrictionsNum;

            addRelationshipsMatrix();

            for (let restrictionsIndex = 0; restrictionsIndex < restrictionsNum; restrictionsIndex++) {
                for (let variablesIndex = 0; variablesIndex < variablesNum; variablesIndex++)
                    document.getElementById(`row${restrictionsIndex}Column${variablesIndex}Input`).value =
                        data[2 + restrictionsIndex * (2 + variablesNum) + variablesIndex];
                document.getElementById(`row${restrictionsIndex}RightInput`).value =
                    data[2 + restrictionsIndex * (2 + variablesNum) + variablesNum + 1];
                document.getElementById(`row${restrictionsIndex}Select`).value =
                    data[2 + restrictionsIndex * (2 + variablesNum) + variablesNum];
            }
        },
        false,
    );

    if (file) {
        reader.readAsText(file);
    }
}

async function createEquationInput(elementId, showcaseId,
								   inequality = false,
								   inequalitySignId = null,
								   inequalityRightId = null) {
    let functionEquation = document.createElement('input');
    functionEquation.id = elementId;
    functionEquation.type = "type";
    functionEquation.autocapitalize = "off";
    functionEquation.spellcheck = false;
    functionEquation.addEventListener("change", () => {
        parse(elementId, showcaseId, inequality, inequalitySignId, inequalityRightId)
    });
        functionEquation.addEventListener("keydown", () => {
        parse(elementId, showcaseId, inequality, inequalitySignId, inequalityRightId)
    });
    functionEquation.className="equation-input";
    return functionEquation;
}

async function addRelationshipsMatrix() {
    let variablesNum = parseInt(document.getElementById('variables_num').value);
    let restrictionsNum = parseInt(document.getElementById('restrictions_num').value);
    if ((isNaN(variablesNum) || variablesNum <= 0 || variablesNum > 50) ||
        (isNaN(restrictionsNum) || restrictionsNum <= 0 || restrictionsNum > 20)){
        return;
    }

    document.getElementById('calculator_body').hidden = false;

    let container = document.getElementById('bottomInputField');

    while (container.hasChildNodes())
        container.removeChild(container.lastChild);

    document.getElementById('last_num').innerHTML = variablesNum;
    document.getElementById('args_rest').style.display = variablesNum < 3 ? 'none' : '';
    document.getElementById('args_rest_dot').style.display = variablesNum < 3 ? 'none' : '';
	document.getElementById('args_first_dot').style.display = variablesNum < 2 ? 'none' : '';
	document.getElementById('last_arg').style.display = variablesNum < 2 ? 'none' : '';

    let functionInput = document.createElement('div');

    let functionName = document.createElement('span');
    functionName.innerHTML='F(x)';
    functionName.style.paddingRight='5px';
    functionInput.appendChild(functionName);
    functionInput.innerHTML += "=";
    functionInput.style.marginBottom = '10px';
    functionInput.className='inputMatrix';
    functionInput.style.gridTemplateColumns = `max-content max-content 1fr`;

    functionInput.appendChild(await createEquationInput("functionEquation", "functionShowcase"));

    container.appendChild(functionInput);
	parse("functionEquation", "functionShowcase",
		false, null, null);
	console.log(document.getElementById(`functionShowcase`));

	document.getElementById('figure-bracket').style.gridRow = `1 / ${variablesNum}`;

    let matrixInput = document.createElement('div');
    matrixInput.className='inputMatrix';
    container.appendChild(matrixInput);

    matrixInput.style.gridTemplateColumns = ` 1fr max-content max-content`;

    let systemShowcase = document.getElementById('systemShowcase');
    while (systemShowcase.hasChildNodes())
        systemShowcase.removeChild(systemShowcase.lastChild);
    for (let row_index = 0; row_index < restrictionsNum; row_index++) {
        await convertStringToMath(
            `<mrow id="matrix${row_index}Showcase" style="margin-bottom: 10px; margin-top: 10px"></mrow>`).
            then(s => {systemShowcase.innerHTML += s;});
        matrixInput.appendChild(await createEquationInput(`matrix${row_index}Equation`,
                                                          `matrix${row_index}Showcase`,
														  true,
														  `row${row_index}Select`,
														  `row${row_index}RightInput`));
        let operatorCell = document.createElement('div');
        let selectOperatorList = document.createElement('select');
		selectOperatorList.disabled = true;
		selectOperatorList.addEventListener("change", () => {
        	parse(`matrix${row_index}Equation`,
				`matrix${row_index}Showcase`,
				true,
				`row${row_index}Select`,
				`row${row_index}RightInput`);
    	});
		selectOperatorList.addEventListener("keydown", () => {
        	parse(`matrix${row_index}Equation`,
				`matrix${row_index}Showcase`,
				true,
				`row${row_index}Select`,
				`row${row_index}RightInput`);
    	});
        selectOperatorList.id = `row${row_index}Select`;

        const operatorArray = ["≤", /*"=", "≥"*/];
        for (let operatorIndex = 0; operatorIndex < operatorArray.length; operatorIndex++) {
            let option = document.createElement('option');
            option.value = operatorArray[operatorIndex];
            option.text = operatorArray[operatorIndex];
            selectOperatorList.appendChild(option);
        }
        operatorCell.appendChild(selectOperatorList);
        matrixInput.appendChild(operatorCell);

        let matrixRightCell = document.createElement('div');
        matrixRightCell.style.width='max-content';

        let matrixRightCellInput = document.createElement('input');
        matrixRightCellInput.type = 'number';
        matrixRightCellInput.step = 'any';
        matrixRightCellInput.id = `row${row_index}RightInput`;
        matrixRightCellInput.placeholder='0';
		matrixRightCellInput.addEventListener("change", () => {
        	parse(`matrix${row_index}Equation`,
				`matrix${row_index}Showcase`,
				true,
				`row${row_index}Select`,
				`row${row_index}RightInput`);
    	});
		matrixRightCellInput.addEventListener("keydown", () => {
        	parse(`matrix${row_index}Equation`,
				`matrix${row_index}Showcase`,
				true,
				`row${row_index}Select`,
				`row${row_index}RightInput`);
    	});
        matrixRightCell.appendChild(matrixRightCellInput);
        matrixInput.appendChild(matrixRightCell);
		parse(`matrix${row_index}Equation`,
			`matrix${row_index}Showcase`,
			true,
			`row${row_index}Select`,
			`row${row_index}RightInput`);
		console.log(document.getElementById(`matrix${row_index}Showcase`));
    }
	//
    // let signInput = document.createElement('div');
    // signInput.className='inputMatrix';
    // signInput.style.gridTemplateColumns = `repeat(${variablesNum}, max-content)`;
    // signInput.style.gap='25px';
    // signInput.style.marginTop="10px";
    // container.appendChild(signInput);
	//
    // for (let variable_index = 0; variable_index < variablesNum; variable_index++) {
    //     let signVariableContainer = document.createElement('div');
	//
    //     let variableName = document.createElement('span');
	// 	variableName.id = `x${variable_index + 1}Name`;
	// 	variableName.value = `x${variable_index + 1}`;
    //     variableName.innerHTML=`x${variable_index + 1}`;
    //     signVariableContainer.appendChild(variableName);
	//
    //     let operatorCell = document.createElement('div');
    //     let selectOperatorList = document.createElement('select');
	// 	await convertStringToMath(
    //         `<mrow id="variable${variable_index}SignShowcase" style="margin-bottom: 10px; margin-top: 10px"></mrow>`).
    //         then(s => {systemShowcase.innerHTML += s;});
    //     selectOperatorList.id = `variable${variable_index}Select`;
	// 	selectOperatorList.disabled = true;
	// 	selectOperatorList.style.margin = '5px';
	// 	selectOperatorList.addEventListener("change", () => {
    //     	parse(`x${variable_index + 1}Name`,
	// 			`variable${variable_index}SignShowcase`,
	// 			true,
	// 			`variable${variable_index}Select`,
	// 			`variable${variable_index}SignValue`);
    // 	});
    //     const operatorArray = ["≥"/*, "=", "≤"*/];
    //     for (let operatorIndex = 0; operatorIndex < operatorArray.length; operatorIndex++) {
    //         let option = document.createElement('option');
    //         option.value = operatorArray[operatorIndex];
    //         option.text = operatorArray[operatorIndex];
    //         selectOperatorList.appendChild(option);
    //     }
    //     operatorCell.appendChild(selectOperatorList);
    //     signVariableContainer.appendChild(operatorCell);
	//
    //     let variableValue = document.createElement('span');
	// 	variableValue.id=`variable${variable_index}SignValue`;
	// 	variableValue.value=0;
    //     variableValue.innerHTML='0';
    //     signVariableContainer.appendChild(variableValue);
	//
	// 	parse(`x${variable_index + 1}Name`,
	// 			`variable${variable_index}SignShowcase`,
	// 			true,
	// 			`variable${variable_index}Select`,
	// 			`variable${variable_index}SignValue`);
    //     signInput.appendChild(signVariableContainer);
    // }
}

async function toMathMl(expression) {
	let str = ``;
	if (expression.Binary) {
		if (expression.Binary.operator === '/') {
			str += `<mfrac>`
			if (expression.Binary.left.Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.Binary.left.Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.Binary.left).then(e => str = str +  e)
			if (expression.Binary.right.Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.Binary.right.Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.Binary.right).then(e => str = str +  e)
			str += `</mfrac>`
		} else {
			str += `<mrow>`
			if (expression.Binary.left.Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.Binary.left.Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.Binary.left).then(e => str = str +  e)
			str += `<mo>`
			str += expression.Binary.operator === '*' ? '×' : expression.Binary.operator;
			str += `</mo>`
			if (expression.Binary.right.Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.Binary.right.Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.Binary.right).then(e => str = str +  e)
			str += `</mrow>`
		}
	} else if (expression.Unary) {
		str += `<mo>`
		str = str + expression.Unary.operator
		str += `</mo>`
		if (expression.Unary.expression.Expression) {
			str += `<mrow>`
			str += `<mo>(</mo>`
			await toMathMl(expression.Unary.expression.Expression).then(e => str = str +  e)
			str += `<mo>)</mo>`
			str += `</mrow>`
		} else
			await toMathMl(expression.Unary.expression).then(e => str = str +  e)
	} else if (expression.FunctionCall) {
		if (expression.FunctionCall.name === `factorial`) {
			if (expression.FunctionCall.args[0].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str = str + e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[0]).then(e => str = str +  e)
			str += `<mo>`
			str = str + '!'
			str += `</mo>`
		} else if (expression.FunctionCall.name === `sqrt`) {
			if (expression.FunctionCall.args.length === 1 ||
			   (expression.FunctionCall.args[expression.FunctionCall.args.length - 1].Number === '2')) {
				str += `<msqrt>`
				str += `<mrow>`
				if (expression.FunctionCall.args[0].Expression) {
					str += `<mrow>`
					str += `<mo>(</mo>`
					await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str += e)
					str += `<mo>)</mo>`
					str += `</mrow>`
				} else
					await toMathMl(expression.FunctionCall.args[0]).then(e => str += e)
				str += `</mrow>`
				str += `</msqrt>`
			} else {
				str += `<mrow class="absolute"><msub class="down"><mi></mi>`
				if (expression.FunctionCall.args[expression.FunctionCall.args.length - 1].Expression) {
					str += `<mrow>`
					str += `<mo>(</mo>`
					await toMathMl(expression.FunctionCall.args[expression.FunctionCall.args.length - 1].Expression).then(e => str += e)
					str += `<mo>)</mo>`
					str += `</mrow>`
				} else
					await toMathMl(expression.FunctionCall.args[expression.FunctionCall.args.length - 1]).then(e => str += e)
				str += `</msub>`
				str += `<msqrt><mrow>`
				if (expression.FunctionCall.args[0].Expression) {
						str += `<mrow>`
						str += `<mo>(</mo>`
						await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str += e)
						str += `<mo>)</mo>`
						str += `</mrow>`
					} else
						await toMathMl(expression.FunctionCall.args[0]).then(e => str += e)

				str += `</mrow>`
				str += `</msqrt></mrow>`
			}
		}  else if (expression.FunctionCall.name === `pow`) {
			str += `<mrow><msup>`
			if (expression.FunctionCall.args[0].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[0]).then(e => str = str +  e)
			if (expression.FunctionCall.args[1].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[1].Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[1]).then(e => str = str +  e)
			str += `</msub></mrow>`
		} else if (expression.FunctionCall.name === `log`) {
			str += `<mrow><msub>`
			str += `<mi>${expression.FunctionCall.name}</mi>`
			if (expression.FunctionCall.args[1].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[1].Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[1]).then(e => str = str +  e)
			str += `</msub>`
			if (expression.FunctionCall.args[0].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[0]).then(e => str = str +  e)
			str += `</mrow>`
		} else if (expression.FunctionCall.name === `min` || expression.FunctionCall.name === `max` ||
				   expression.FunctionCall.name === `equals`) {
			str += `<mrow>`
			if (expression.FunctionCall.args[0].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[0].Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[0]).then(e => str = str +  e)
			str += `<mo>`
			if (expression.FunctionCall.name === `min`)
				str += '≤'
			if (expression.FunctionCall.name === `max`)
				str += '≥'
			if (expression.FunctionCall.name === `equals`)
				str += '='
			str += `</mo>`
			if (expression.FunctionCall.args[1].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[1].Expression).then(e => str = str +  e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[1]).then(e => str = str +  e)
			str += `</mrow>`
		} else {
			str += `<mrow>`
			str += "<mo>" + expression.FunctionCall.name + "</mo>"
			for (let arg_index = 0; arg_index < expression.FunctionCall.args.length - 1; arg_index++) {
				if (expression.FunctionCall.args[arg_index].Expression) {
					str += `<mrow>`
					str += `<mo>(</mo>`
					await toMathMl(expression.FunctionCall.args[arg_index].Expression).then(e => str += e)
					str += `<mo>)</mo>`
					str += `</mrow>`
				} else
					await toMathMl(expression.FunctionCall.args[arg_index]).then(e => str += e)
				if (arg_index < expression.FunctionCall.args.length - 2) {
					str += `<mo>`
					str += ','
					str += `</mo>`
				}
			}
			if (expression.FunctionCall.args[expression.FunctionCall.args.length - 1].Expression) {
				str += `<mrow>`
				str += `<mo>(</mo>`
				await toMathMl(expression.FunctionCall.args[expression.FunctionCall.args.length - 1].Expression).then(e => str += e)
				str += `<mo>)</mo>`
				str += `</mrow>`
			} else
				await toMathMl(expression.FunctionCall.args[expression.FunctionCall.args.length - 1]).then(e => str += e)
			str += `</mrow>`
		}
	} else if (expression.Identifier) {
        if (expression.Identifier === `pi` || expression.Identifier.length < 2) {
            str += `<mi>`
            str += expression.Identifier === `pi` ? 'π' : expression.Identifier
            str += `</mi>`
        } else {
            str += `<msub>`;
            str += `<mi>${expression.Identifier[0]}</mi>`;
            str += `<mi>${expression.Identifier.slice(1)}</mi>`;
            str += `</msub>`;
        }
	} else if (expression.Number) {
		str += `<mn>`
		str += expression.Number
		str += `</mn>`
	}
	return str
}

const convertStringToMath = async htmlString => {
	const parser = new DOMParser();
	const html = parser.parseFromString(htmlString, 'text/html');
	return html.body.innerHTML;
}