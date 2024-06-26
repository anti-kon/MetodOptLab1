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

async function addRelationshipsMatrix() {
    let variablesNum = parseInt(document.getElementById('variables_num').value);
    let restrictionsNum = parseInt(document.getElementById('restrictions_num').value);
    if ((isNaN(variablesNum) || variablesNum <= 0 || variablesNum > 50) ||
        (isNaN(restrictionsNum) || restrictionsNum <= 0 || restrictionsNum > 20)){
        return;
    }

    let container = document.getElementById('inputField');

    while (container.hasChildNodes())
        container.removeChild(container.lastChild);

    let functionInput = document.createElement('div');
    let functionName = document.createElement('span');
    functionName.innerHTML='F(x)';
    functionName.style.paddingRight='5px';
    functionInput.appendChild(functionName);
    functionInput.innerHTML += "=";
    functionInput.style.marginBottom = '10px';

    functionInput.className='inputMatrix';
    functionInput.style.gridTemplateColumns = `repeat(${2 * variablesNum + 1}, max-content)`
    container.appendChild(functionInput);

    for (let variable_index = 0; variable_index < variablesNum; variable_index++) {
        let matrixCell = document.createElement('div');
        matrixCell.style.width='max-content';
        matrixCell.style.paddingRight='5px';

        let matrixCellInput = document.createElement('input');
        matrixCellInput.type = 'number';
        matrixCellInput.step = 'any';
        matrixCellInput.id = `function${variable_index}Input`;
        matrixCellInput.placeholder='0';
        matrixCell.appendChild(matrixCellInput);

        matrixCell.innerHTML += `x${(variable_index + 1)}`;
        functionInput.appendChild(matrixCell);

        if (variable_index < variablesNum - 1) {
            let operatorCell = document.createElement('div');
            operatorCell.innerHTML = '+';
            functionInput.appendChild(operatorCell);
        }
    }

    let matrixInput = document.createElement('div');
    matrixInput.className='inputMatrix';
    container.appendChild(matrixInput);

    matrixInput.style.gridTemplateColumns = `repeat(${2 * variablesNum + 1}, max-content)`;

    for (let row_index = 0; row_index < restrictionsNum; row_index++) {
        for (let column_index = 0; column_index < variablesNum; column_index++) {
            let matrixCell = document.createElement('div');
            matrixCell.style.width='max-content';
            matrixCell.style.paddingRight='5px';

            let matrixCellInput = document.createElement('input');
            matrixCellInput.type = 'number';
            matrixCellInput.step = 'any';
            matrixCellInput.id = `row${row_index}Column${column_index}Input`;
            matrixCellInput.placeholder='0';
            matrixCell.appendChild(matrixCellInput);

            matrixCell.innerHTML += `x${(column_index + 1)}`;
            matrixInput.appendChild(matrixCell);

            if (column_index < variablesNum - 1) {
                let operatorCell = document.createElement('div');
                operatorCell.innerHTML = '+';
                matrixInput.appendChild(operatorCell);
            }
        }
        let operatorCell = document.createElement('div');
        let selectOperatorList = document.createElement('select');
        selectOperatorList.id = `row${row_index}Select`;

        const operatorArray = ["≤", "=", "≥"];
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
        matrixRightCell.appendChild(matrixRightCellInput);
        matrixInput.appendChild(matrixRightCell);
    }

    let signInput = document.createElement('div');
    signInput.className='inputMatrix';
    signInput.style.gridTemplateColumns = `repeat(${variablesNum}, max-content)`;
    signInput.style.gap='25px';
    signInput.style.marginTop="10px";
    container.appendChild(signInput);

    for (let variable_index = 0; variable_index < variablesNum; variable_index++) {
        let signVariableContainer = document.createElement('div');

        let variableName = document.createElement('span');
        variableName.innerHTML=`x${variable_index + 1}`;
        signVariableContainer.appendChild(variableName);

        let operatorCell = document.createElement('div');
        let selectOperatorList = document.createElement('select');
        selectOperatorList.id = `variable${variable_index}Select`;
        const operatorArray = ["≤", "=", "≥"];
        for (let operatorIndex = 0; operatorIndex < operatorArray.length; operatorIndex++) {
            let option = document.createElement('option');
            option.value = operatorArray[operatorIndex];
            option.text = operatorArray[operatorIndex];
            selectOperatorList.appendChild(option);
        }
        operatorCell.appendChild(selectOperatorList);
        signVariableContainer.appendChild(operatorCell);

        let variableValue = document.createElement('span');
        variableValue.innerHTML='0';
        signVariableContainer.appendChild(variableValue);

        signInput.appendChild(signVariableContainer);
    }
}