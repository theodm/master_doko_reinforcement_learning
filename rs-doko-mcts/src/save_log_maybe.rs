use std::fs;
use chrono::Local;
use std::fs::File;
use std::io::Write;

/// Diese Methode schreibt das ganze in eine eigene Datei
/// im definiertenUnterordner. Nur wenn gesetzt. Und stellt auch sicher,
/// dass es nicht mehr als 50 Dateien gibt. In diesem Fall werden alte Dateien
/// gelöscht. Mit deutschem Zeitstempel wird gespeichert.
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

        // multi_progress
        //     .println(format!("Log saved to {}", filename))
        //     .unwrap();

        let mut entries: Vec<_> = fs::read_dir(folder)?
            .filter_map(|res| res.ok()) 
            .filter_map(|entry| {
                let path = entry.path();
                // Nur Dateien berücksichtigen
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

        while entries.len() > 50 {
            if let Some(oldest) = entries.first() {
                let _ = fs::remove_file(oldest);
            }
            entries.remove(0);
        }
    }

    Ok(())
}