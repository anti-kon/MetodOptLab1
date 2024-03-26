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

    let container = document.getElementById('inputMatrix');

    while (container.hasChildNodes())
        container.removeChild(container.lastChild);

    container.style.gridTemplateColumns = `repeat(${2 * variablesNum + 1}, max-content)`;

    for (let row_index = 0; row_index < restrictionsNum; row_index++) {
        for (let column_index = 0; column_index < variablesNum; column_index++) {
            let matrixCell = document.createElement('div');
            matrixCell.style.width='max-content';

            let matrixCellInput = document.createElement('input');
            matrixCellInput.type = 'number';
            matrixCellInput.step = 'any';
            matrixCellInput.id = `row${row_index}Column${column_index}Input`;
            matrixCellInput.placeholder='0';
            matrixCell.appendChild(matrixCellInput);

            matrixCell.innerHTML += `x${(column_index + 1)}`;
            container.appendChild(matrixCell);

            if (column_index < variablesNum - 1) {
                let operatorCell = document.createElement('div');
                operatorCell.innerHTML = '+';
                container.appendChild(operatorCell);
            }
        }
        let operatorCell = document.createElement('div');
        let selectOperatorList = document.createElement('select');
        selectOperatorList.id = `row${row_index}Select`;

        const operatorArray = ["=", ">","<"];
        for (let operatorIndex = 0; operatorIndex < operatorArray.length; operatorIndex++) {
            let option = document.createElement('option');
            option.value = operatorArray[operatorIndex];
            option.text = operatorArray[operatorIndex];
            selectOperatorList.appendChild(option);
        }
        operatorCell.appendChild(selectOperatorList);
        container.appendChild(operatorCell);

        let matrixRightCell = document.createElement('div');
        matrixRightCell.style.width='max-content';

        let matrixRightCellInput = document.createElement('input');
        matrixRightCellInput.type = 'number';
        matrixRightCellInput.step = 'any';
        matrixRightCellInput.id = `row${row_index}RightInput`;
        matrixRightCellInput.placeholder='0';
        matrixRightCell.appendChild(matrixRightCellInput);
        container.appendChild(matrixRightCell);
    }
}