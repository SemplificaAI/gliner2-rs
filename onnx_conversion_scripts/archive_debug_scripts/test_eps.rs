use ort::execution_providers::*;
fn main() {
    let _ = QNNExecutionProvider::default();
    let _ = OpenVINOExecutionProvider::default();
    let _ = CoreMLExecutionProvider::default();
    let _ = XNNPACKExecutionProvider::default();
}
