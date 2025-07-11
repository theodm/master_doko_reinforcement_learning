use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::sync::mpsc::{self, Sender, Receiver};
use std::thread;
use std::path::PathBuf;
use std::time::Duration;
use indicatif::{ProgressBar, ProgressStyle};
use csv::WriterBuilder;

enum CSVWriterMessage {
    Row(Vec<String>),
    Finish,
}


pub struct CSVWriterThread {
    tx: Sender<CSVWriterMessage>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Clone for CSVWriterThread {
    fn clone(&self) -> Self {
        return CSVWriterThread {
            tx: self.tx.clone(),
            handle: None,
        };
    }
}

impl CSVWriterThread {
    pub fn new(
        path: PathBuf,
        headers: &[&str],
        expected_rows: Option<u64>,
        append_only: bool,
    ) -> CSVWriterThread {
        let headers_owned: Vec<String> = headers.iter().map(|s| s.to_string()).collect();

        let (tx, rx): (Sender<CSVWriterMessage>, Receiver<CSVWriterMessage>) = mpsc::channel();

        let handle = thread::spawn(move || {
            let file = if append_only {
                OpenOptions::new()
                    .write(true)
                    .append(true)
                    .create(true)
                    .open(&path)
                    .expect("Konnte CSV-Datei nicht im Append-Modus öffnen.")
            } else {
                File::create(&path)
                    .expect("Konnte CSV-Datei nicht erstellen.")
            };

            let writer_buf = BufWriter::new(file);

            let mut writer = WriterBuilder::new()
                .has_headers(!append_only)
                .from_writer(writer_buf);

            if !append_only {
                writer.write_record(headers_owned).expect("Konnte Header nicht schreiben");
                writer.flush().unwrap();
            }

            let pb = if let Some(total) = expected_rows {
                let pb = ProgressBar::new(total);

                pb
                    .enable_steady_tick(Duration::from_secs(1));

                pb.set_style(
                    ProgressStyle::with_template(
                        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})"
                    )
                        .unwrap()
                        .progress_chars("#>-")
                );

                Some(pb)
            } else {
                None
            };

            while let Ok(message) = rx.recv() {
                match message {
                    CSVWriterMessage::Row(row) => {
                        writer.write_record(&row).expect("Konnte Zeile nicht schreiben");
                        writer.flush().unwrap();

                        if let Some(ref pb) = pb {
                            pb.inc(1);
                        }
                    }
                    CSVWriterMessage::Finish => {
                        break;
                    }
                }
            }

            if let Some(pb) = pb {
                pb.finish();
            }
        });

        CSVWriterThread { tx, handle: Some(handle) }
    }

    pub fn write_row<S: Into<String>>(&self, row: Vec<S>) {
        let row_as_strings: Vec<String> = row.into_iter().map(|s| s.into()).collect();

        self.tx
            .send(CSVWriterMessage::Row(row_as_strings))
            .expect("Konnte Zeile nicht an CSV-Thread senden");
    }

    pub fn finish(self) {
        self.tx
            .send(CSVWriterMessage::Finish)
            .expect("Konnte Finish-Nachricht nicht an CSV-Thread senden");

        let _ = self.handle.unwrap().join();
    }
}
