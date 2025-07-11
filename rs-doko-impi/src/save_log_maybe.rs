use std::fs;
use chrono::Local;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use indicatif::MultiProgress;

pub fn save_log_maybe(
    log: Option<String>,
    folder: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(text) = log {
        fs::create_dir_all(folder)?;

        let now = Local::now();
        let timestamp = now.format("%d.%m.%Y_%H-%M-%S").to_string();

        let filename = format!("{}/log_{}.txt", folder, timestamp);

        let mut file = File::create(&filename)?;
        file.write_all(text.as_bytes())?;

        let mut entries: Vec<_> = fs::read_dir(folder)?
            .filter_map(|res| res.ok()) 
            .filter_map(|entry| {
                let path = entry.path();
                match fs::metadata(&path) {
                    Ok(metadata) if metadata.is_file() => Some(path),
                    _ => None,
                }
            })
            .collect();

        entries.sort_by_key(|path| {
            fs::metadata(path)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });

        while entries.len() > 700 {
            if let Some(oldest) = entries.first() {
                let _ = fs::remove_file(oldest);
            }
            entries.remove(0);
        }
    }

    Ok(())
}