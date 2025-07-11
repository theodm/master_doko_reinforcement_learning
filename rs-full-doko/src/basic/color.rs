use enumset::EnumSetType;

#[derive(Debug, EnumSetType)]
pub enum FdoColor {
    Trump,

    // Es gibt die Farbe nur in Sonderspielen! Im Normalspiel
    // ist Karo immer Trumpf.
    Diamond,
    Heart,
    Spade,
    Club,
}