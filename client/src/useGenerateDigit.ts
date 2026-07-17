import { useMutation } from "@tanstack/react-query";
import { fetchImage } from "./api";

export function useGenerateDigit() {
    return useMutation({
        mutationFn: fetchImage,
    });
}
