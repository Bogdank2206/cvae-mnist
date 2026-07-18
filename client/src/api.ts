import axios from "axios";

export interface GenerateResponse {
    image: string;
}

const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL ?? "http://localhost:8000",
});

export async function fetchImage(label: number): Promise<GenerateResponse> {
    const {
        data: { image },
    } = await api.post<GenerateResponse>("/generate", { label });
    return { image };
}
