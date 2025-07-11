use serde::{Deserialize, Serialize};
use rs_game_utils::bit_flag::Bitflag;
use strum::EnumCount;
use crate::announcement::announcement::FdoAnnouncement;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FdoAnnouncementSet(Bitflag<{ FdoAnnouncement::COUNT }>);

impl FdoAnnouncementSet {
    pub fn new() -> FdoAnnouncementSet {
        FdoAnnouncementSet(Bitflag::new())
    }

    pub fn from_vec(announcements: Vec<FdoAnnouncement>) -> FdoAnnouncementSet {
        let mut set = FdoAnnouncementSet::new();

        for announcement in announcements {
            set.add(announcement);
        }

        set
    }

    /// Gibt ein Set zurück, dass alle Ansagen enthält, die höher sind als die übergebene Ansage.
    pub fn all_higher_than(lowest_announcement: Option<FdoAnnouncement>) -> FdoAnnouncementSet {
        let mut set = FdoAnnouncementSet::new();

        match lowest_announcement {
            None => {},
            Some(announcement) => match announcement  {
                FdoAnnouncement::ReContra => {
                    set.add(FdoAnnouncement::ReContra);
                }
                FdoAnnouncement::No90 => {
                    set.add(FdoAnnouncement::ReContra);
                    set.add(FdoAnnouncement::No90);
                }
                FdoAnnouncement::No60 => {
                    set.add(FdoAnnouncement::ReContra);
                    set.add(FdoAnnouncement::No90);
                    set.add(FdoAnnouncement::No60);
                }
                FdoAnnouncement::No30 => {
                    set.add(FdoAnnouncement::ReContra);
                    set.add(FdoAnnouncement::No90);
                    set.add(FdoAnnouncement::No60);
                    set.add(FdoAnnouncement::No30);
                }
                FdoAnnouncement::Black => {
                    set.add(FdoAnnouncement::ReContra);
                    set.add(FdoAnnouncement::No90);
                    set.add(FdoAnnouncement::No60);
                    set.add(FdoAnnouncement::No30);
                    set.add(FdoAnnouncement::Black);
                }
                FdoAnnouncement::CounterReContra => {
                    set.add(FdoAnnouncement::CounterReContra);
                }
                _ => {}
            }
        }

        set
    }

    pub fn add(&mut self, announcement: FdoAnnouncement) {
        self.0.add(announcement as u64);
    }

    pub fn contains(&self, announcement: FdoAnnouncement) -> bool {
        self.0.contains(announcement as u64)
    }

    pub fn remove(&mut self, announcement: FdoAnnouncement) {
        self.0.remove(announcement as u64);
    }

    pub fn len(&self) -> usize {
        self.0.number_of_ones() as usize
    }

}