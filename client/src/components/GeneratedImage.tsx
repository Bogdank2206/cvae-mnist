import { ImageCard } from "./ImageCard";

interface GeneratedImageProps {
    image: string | null;
    isLoading: boolean;
    isError: boolean;
    label: number | null;
}

export function GeneratedImage({ image, isLoading, isError, label }: GeneratedImageProps) {
    return (
        <div className="flex flex-col items-center gap-4">
            <h2 className="text-lg font-medium text-slate-300">Результат</h2>

            <ImageCard
                image={image}
                isLoading={isLoading}
                isError={isError}
                label={label}
            />
        </div>
    );
}
