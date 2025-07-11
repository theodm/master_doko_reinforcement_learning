


#[cfg(test)]
mod tests {
    use tch::{IndexOp, Tensor};

    #[test]
    fn tensor_test2() {
        let x = tch::Tensor::from_slice(&[
            // Karte 1
            1, 2, 3,
            // Karte 2
            3, 4, 5,
            // Karte 3
            5, 6, 7,
            // Karte 4
            7, 8, 9,
            // Karte 5
            9, 10, 11,
            // Karte 6
            11, 12, 13,
            // Karte 7
            13, 14, 15,
            // Karte 8
            15, 16, 17,
            // Spieler Trick 1
            7, 18,
            // Spieler Trick 2
            19, 20,

            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
            .view([2, -1])
            .to_kind(tch::Kind::Float);

        println!("{}", x);

        let card_size = 3;
        let player_size = 2;

        let cards = x
            .i((.., 0 .. 8 * card_size))
            .view([2, -1,  card_size]);
        let players = x
            .i((.., 8*card_size.. 8*card_size + 2 * player_size))
            .view([2, -1, player_size]);

        // cards in 4er Blöcken
        let cards = cards.view([2, 2, 4 * card_size]);
        println!("{}", cards);

        // Und Spieler passend
        let players = players.view([2, 2, player_size]);

        println!("{}", players);

        // Nun Cat?
        let x = Tensor::cat(&[cards, players], 2);

        println!("{}", x);

    }

    #[test]
    fn tensor_test() {
        let x = tch::Tensor::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            .view([2, 4, 3])
            .to_kind(tch::Kind::Float);

        let x = x.permute(&[0, 2, 1]);

        println!("{}", x);

        let y = x.avg_pool1d(
            &[4],
            &[4],
            &[0],
            false,
            false
        );

        println!("{}", y);
    }
}