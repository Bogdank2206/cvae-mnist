export interface GenerateResponse {
    image: string;
}

export async function fetchImage(label: number): Promise<GenerateResponse> {
    await new Promise((resolve) => setTimeout(resolve, 400));

    return { image: MOCK_IMAGES[label % 10] };
}

const MOCK_IMAGES = [
    "data:image/svg+xml," +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"><rect width="28" height="28" fill="#000"/><text x="14" y="20" text-anchor="middle" fill="#fff" font-size="20" font-family="serif">0</text></svg>',
        ),
    "data:image/svg+xml," +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"><rect width="28" height="28" fill="#000"/><text x="14" y="20" text-anchor="middle" fill="#fff" font-size="20" font-family="serif">1</text></svg>',
        ),
    "data:image/svg+xml," +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"><rect width="28" height="28" fill="#000"/><text x="14" y="20" text-anchor="middle" fill="#fff" font-size="20" font-family="serif">2</text></svg>',
        ),
    "data:image/svg+xml," +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"><rect width="28" height="28" fill="#000"/><text x="14" y="20" text-anchor="middle" fill="#fff" font-size="20" font-family="serif">3</text></svg>',
        ),
    "data:image/svg+xml," +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"><rect width="28" height="28" fill="#000"/><text x="14" y="20" text-anchor="middle" fill="#fff" font-size="20" font-family="serif">4</text></svg>',
        ),
    "data:image/svg+xml," +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"><rect width="28" height="28" fill="#000"/><text x="14" y="20" text-anchor="middle" fill="#fff" font-size="20" font-family="serif">5</text></svg>',
        ),
    "data:image/svg+xml," +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"><rect width="28" height="28" fill="#000"/><text x="14" y="20" text-anchor="middle" fill="#fff" font-size="20" font-family="serif">6</text></svg>',
        ),
    "data:image/svg+xml," +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"><rect width="28" height="28" fill="#000"/><text x="14" y="20" text-anchor="middle" fill="#fff" font-size="20" font-family="serif">7</text></svg>',
        ),
    "data:image/svg+xml," +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"><rect width="28" height="28" fill="#000"/><text x="14" y="20" text-anchor="middle" fill="#fff" font-size="20" font-family="serif">8</text></svg>',
        ),
    "data:image/svg+xml," +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"><rect width="28" height="28" fill="#000"/><text x="14" y="20" text-anchor="middle" fill="#fff" font-size="20" font-family="serif">9</text></svg>',
        ),
];
