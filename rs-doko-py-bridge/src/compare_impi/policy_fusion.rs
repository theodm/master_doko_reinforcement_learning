use std::cmp::Reverse;
use std::mem::transmute;
use async_trait::async_trait;
use rs_full_doko::action::action::FdoAction;
use strum::{EnumCount, IntoEnumIterator};
use rs_full_doko::action::allowed_actions::FdoAllowedActions;

#[async_trait]
pub trait PolicyFusionFn: Send + Sync {
    async fn fuse(
        &self,
        moves_to_visits: Vec<[usize; { FdoAction::COUNT }]>,
        moves_to_values: Vec<[f32; { FdoAction::COUNT }]>,
        allowed_actions: FdoAllowedActions
    ) -> FdoAction;
}

pub struct PolicyFusionMaxN {

}


#[async_trait]
impl PolicyFusionFn for PolicyFusionMaxN {
    async fn fuse(
        &self,
        moves_to_visits: Vec<[usize; FdoAction::COUNT]>,
        _moves_to_values: Vec<[f32; FdoAction::COUNT]>,
        allowed_actions: FdoAllowedActions,
    ) -> FdoAction {
        // Initialisiere kumulierte Ränge mit 0
        let mut cumulative_ranks = [0u32; FdoAction::COUNT];

        for action in FdoAction::iter() {
            // Setze den Rang für die nicht erlaubten Aktionen auf einen hohen Wert
            if !allowed_actions.contains(action) {
                cumulative_ranks[action.to_index()] = u32::MAX;
            }
        }

        // Für jeden Zustand: Aktion mit den meisten Visits bekommt Rang 1, nächster Rang 2, …
        for state_visits in &moves_to_visits {
            // Paare (Action‑Index, Visits), dann absteigend nach Visits sortieren
            let mut pairs: Vec<(usize, usize)> = state_visits
                .iter()
                .cloned()
                .enumerate()
                .filter(|&(action_idx, _visits)| {
                    // Filtere nur erlaubte Aktionen
                    allowed_actions.contains(FdoAction::from_index(action_idx))
                })
                .collect();

            pairs.sort_by_key(|&(_idx, visits)| Reverse(visits));


            // Vergabe der Ränge (1-basiert) und aufsummieren
            for (rank_zero_based, &(action_idx, _visits)) in pairs.iter().enumerate() {
                // rank = rank_zero_based + 1
                cumulative_ranks[action_idx] += (rank_zero_based as u32) + 1;
            }
        }

        // Finde die Aktion mit dem geringsten kumulierten Rang
        let best_index = cumulative_ranks
            .iter()
            .enumerate()
            .min_by_key(|&(_idx, &sum_rank)| sum_rank)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        FdoAction::from_index(best_index)
    }
}


pub struct PolicyFusionAverageStrategy {
}

#[async_trait]
impl PolicyFusionFn for PolicyFusionAverageStrategy {
    async fn fuse(
        &self,
        moves_to_visits: Vec<[usize; { FdoAction::COUNT }]>,
        _moves_to_values: Vec<[f32; { FdoAction::COUNT }]>,
        _allowed_actions: FdoAllowedActions,
    ) -> FdoAction {
        let num_states = moves_to_visits.len();

        let mut sum_probabilities = [0.0f32; { FdoAction::COUNT }];

        for state_visits in &moves_to_visits {
            let total_visits_in_state = state_visits.iter().sum::<usize>();

            for action_index in 0..FdoAction::COUNT {
                let visits_for_action = state_visits[action_index];
                let probability = visits_for_action as f32 / total_visits_in_state as f32;
                sum_probabilities[action_index] += probability;
            }
        }

        // Berechne die durchschnittliche Strategie π_avg(a)
        // Hinweis: Die Formel ist π_avg(a) = (1/|S|) * Σ π(s,a).
        // Da wir die Summe der Wahrscheinlichkeiten haben, müssen wir sie nicht unbedingt
        // durch num_states teilen, um das Maximum zu finden, aber es entspricht der Definition.
        // Wir suchen direkt das Maximum in sum_probabilities, da der Faktor 1/|S| das Maximum nicht ändert.

        // for action_index in 0..FdoAction::COUNT {
        //     // Nun geteilt
        //     println!("Action {:?}: Durchschnittliche Wahrscheinlichkeit = {}", FdoAction::from_index(action_index), sum_probabilities[action_index] / num_states as f32);
        // }

        let (best_action_index, _) = sum_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, &val_a), (_, &val_b)| {
                val_a.partial_cmp(&val_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, &0.0f32));

        FdoAction::from_index(best_action_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_full_doko::action::action::FdoAction;

    #[tokio::test]
    async fn test_policy_fusion_maxn_with_explicit_actions() {
        let fusion = PolicyFusionMaxN {};

        let idx_ten   = FdoAction::CardDiamondTen.to_index();
        let idx_nine  = FdoAction::CardDiamondNine.to_index();
        let idx_jack  = FdoAction::CardDiamondJack.to_index();
        let idx_queen = FdoAction::CardDiamondQueen.to_index();

        let mut dummy = [0usize; FdoAction::COUNT];

        let states = vec![
            {
                let mut s = dummy;
                s[idx_ten]   = 1;
                s[idx_nine]  = 2;
                s[idx_jack]  = 4;
                s[idx_queen] = 3;
                s
            },
            {
                let mut s = dummy;
                s[idx_ten]   = 4;
                s[idx_nine]  = 2;
                s[idx_jack]  = 1;
                s[idx_queen] = 3;
                s
            },
            {
                let mut s = dummy;
                s[idx_ten]   = 5;
                s[idx_nine]  = 2;
                s[idx_jack]  = 3;
                s[idx_queen] = 1;
                s
            },
            {
                let mut s = dummy;
                s[idx_ten]   = 3;
                s[idx_nine]  = 2;
                s[idx_jack]  = 1;
                s[idx_queen] = 4;
                s
            },
        ];

        let dummy_values = vec![[0.0; FdoAction::COUNT]; states.len()];
        let result = fusion.fuse(states, dummy_values, FdoAllowedActions::from_vec(
            vec![
                FdoAction::CardDiamondTen,
                FdoAction::CardDiamondNine,
                FdoAction::CardDiamondJack,
                FdoAction::CardDiamondQueen,
            ]
        )).await;

        assert_eq!(
            result,
            FdoAction::CardDiamondTen,
            "Expected CardDiamondTen, got {:?}",
            result
        );
    }


    #[tokio::test]
    async fn test_policy_fusion_average_strategy() {
        let fusion = PolicyFusionAverageStrategy {};

        let a1_idx = FdoAction::CardDiamondTen.to_index();
        let a2_idx = FdoAction::CardDiamondNine.to_index();
        let a3_idx = FdoAction::CardDiamondJack.to_index();
        let a4_idx = FdoAction::CardDiamondQueen.to_index();

        let mut dummy = [0usize; FdoAction::COUNT];

        let states_visits = vec![
            { 
                let mut s = dummy; s[a1_idx] = 1; s[a2_idx] = 2; s[a3_idx] = 4; s[a4_idx] = 3; s
            },
            { 
                let mut s = dummy; s[a1_idx] = 4; s[a2_idx] = 2; s[a3_idx] = 1; s[a4_idx] = 3; s
            },
            { 
                let mut s = dummy; s[a1_idx] = 5; s[a2_idx] = 2; s[a3_idx] = 3; s[a4_idx] = 1; s
            },
            { 
                let mut s = dummy; s[a1_idx] = 3; s[a2_idx] = 2; s[a3_idx] = 1; s[a4_idx] = 4; s
            },
        ];

        let dummy_values = vec![[0.0; FdoAction::COUNT]; states_visits.len()];

        let result = fusion.fuse(states_visits, dummy_values,
            FdoAllowedActions::from_vec(
                vec![
                    FdoAction::CardDiamondTen,
                    FdoAction::CardDiamondNine,
                    FdoAction::CardDiamondJack,
                    FdoAction::CardDiamondQueen,
                ]
            )
        ).await;

        assert_eq!(
            result,
            FdoAction::CardDiamondTen, // a1
            "Expected action a1 (CardDiamondTen), got {:?}",
            result
        );
    }


    #[tokio::test]
    async fn test_policy_fusion_maxn_with_custom_visits() {
        let fusion = PolicyFusionMaxN {};
        let dummy = [0usize; FdoAction::COUNT];
        let visits: Vec<[usize; FdoAction::COUNT]> = vec![
            { let mut s = dummy; s[26] = 20; s },
            { let mut s = dummy; s[26] =  5; s[31] = 15; s },
            { let mut s = dummy; s[26] = 12; s[31] =  8; s },
            { let mut s = dummy; s[26] =  6; s[31] = 14; s },
            { let mut s = dummy; s[26] =  5; s[31] = 15; s },
            { let mut s = dummy; s[26] = 10; s[31] = 10; s },
            { let mut s = dummy; s[26] =  9; s[31] = 11; s },
            { let mut s = dummy; s[31] = 20; s },
            { let mut s = dummy; s[31] = 20; s },
            { let mut s = dummy; s[26] =  4; s[31] = 16; s },
        ];

        let dummy_values = vec![[0.0; FdoAction::COUNT]; visits.len()];
        let result = fusion.fuse(visits, dummy_values,
            FdoAllowedActions::from_vec(
                vec![
                    FdoAction::from_index(26),
                    FdoAction::from_index(31),
                ]
            )
        ).await;

        assert_eq!(
            result,
            FdoAction::from_index(31),
            "Expected action index 31 for PolicyFusionMaxN, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_policy_fusion_average_strategy_with_custom_visits() {
        let fusion = PolicyFusionAverageStrategy {};
        let dummy = [0usize; FdoAction::COUNT];
        let visits: Vec<[usize; FdoAction::COUNT]> = vec![
            { let mut s = dummy; s[26] = 20; s },
            { let mut s = dummy; s[26] =  5; s[31] = 15; s },
            { let mut s = dummy; s[26] = 12; s[31] =  8; s },
            { let mut s = dummy; s[26] =  6; s[31] = 14; s },
            { let mut s = dummy; s[26] =  5; s[31] = 15; s },
            { let mut s = dummy; s[26] = 10; s[31] = 10; s },
            { let mut s = dummy; s[26] =  9; s[31] = 11; s },
            { let mut s = dummy; s[31] = 20; s },
            { let mut s = dummy; s[31] = 20; s },
            { let mut s = dummy; s[26] =  4; s[31] = 16; s },
        ];

        let dummy_values = vec![[0.0; FdoAction::COUNT]; visits.len()];
        let result = fusion.fuse(visits, dummy_values,
            FdoAllowedActions::from_vec(
                vec![
                    FdoAction::from_index(26),
                    FdoAction::from_index(31),
                ]
            )
        ).await;

        assert_eq!(
            result,
            FdoAction::from_index(31),
            "Expected action index 31 for PolicyFusionAverageStrategy, got {:?}",
            result
        );
    }
}

