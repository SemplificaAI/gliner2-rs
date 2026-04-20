fn main() {
    let _ = ort::init().with_name("test").commit();
    let count_lstm = ort::Session::builder().unwrap().commit_from_file("../models/lmo3-gliner2-multi-v1-onnx/onnx/count_lstm_fp16.onnx").unwrap();
    for i in &count_lstm.inputs {
        println!("Input: {}", i.name);
    }
}
