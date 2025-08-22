window.onload = () => {
    var btn = document.getElementById('clr');
    var val = document.getElementById('val');

    //  Prevent showing Negative when there's no input
    if (val.innerHTML.trim() === "" || val.innerHTML.trim() === "{{ result }}") {
        val.innerHTML = "Waiting for input...";
        val.style.backgroundColor = "gray";
        val.style.color = "white";
    } else {
        val.innerHTML === "Positive" ? val.style.backgroundColor = "green" : val.style.backgroundColor = "red";
        val.innerHTML === "Positive" ? val.style.color = "white" : val.style.color = "white";
    }

    // 🧹 Clear button functionality
    btn.onclick = () => {
        val.innerHTML = "Waiting for input...";
        val.style.backgroundColor = "gray";
        val.style.color = "white";
    };
};
