/**
 * Canonical PD runs for the run picker.
 *
 * Static data (name, notes) renders instantly in the UI.
 * Dynamic data (architecture, availability) is hydrated from the backend.
 */

export type ClusterMappingEntry = { path: string; notes: string };

export type RegistryEntry = {
    wandbRunId: string;
    name?: string;
    notes?: string;
    clusterMappings?: ClusterMappingEntry[];
};

const DEFAULT_ENTITY_PROJECT = "goodfire/spd";

export const CANONICAL_RUNS: RegistryEntry[] = [
    {
        name: "GPT2-XL 0.mlp.up (test)",
        wandbRunId: "goodfire/param-decomp/p-73cf27e4",
        notes: "GPT2-XL block 0 MLP up-projection (c_fc), C=8192. Partial autointerp.",
    },
];

/**
 * Formats a wandb run id for display.
 * Shows just the 8-char run id if it's from "goodfire/spd",
 * otherwise shows the full path.
 */
export function formatRunIdForDisplay(wandbRunId: string): string {
    if (wandbRunId.startsWith(`${DEFAULT_ENTITY_PROJECT}/`)) {
        const parts = wandbRunId.split("/");
        return parts[parts.length - 1];
    }
    return wandbRunId;
}
