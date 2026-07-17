import axios from "axios";

export interface GenerateResponse {
    image: string;
}

const api = axios.create({
    baseURL: "http://localhost:8000",
});

export async function fetchImage(label: number): Promise<GenerateResponse> {
    const {
        data: { image },
    } = await api.post<GenerateResponse>("/generate", { label });
    return { image };
}
