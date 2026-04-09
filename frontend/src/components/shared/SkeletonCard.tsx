import styles from './SkeletonCard.module.css';

interface SkeletonCardProps {
  count?: number;
  /** Optional status message shown above the skeleton cards (e.g. "Generating picks...") */
  message?: string;
}

function SingleSkeleton() {
  return (
    <div className={styles.skeleton}>
      <div className={`${styles['skel-block']} ${styles['skel-circle']}`} />
      <div className={styles['skel-lines']}>
        <div
          className={`${styles['skel-block']} ${styles['skel-line']} ${styles.wide}`}
        />
        <div
          className={`${styles['skel-block']} ${styles['skel-line']} ${styles.narrow}`}
        />
        <div
          className={`${styles['skel-block']} ${styles['skel-line']} ${styles.narrow}`}
        />
      </div>
      <div className={`${styles['skel-block']} ${styles['skel-score']}`} />
    </div>
  );
}

export default function SkeletonCard({ count = 5, message }: SkeletonCardProps) {
  return (
    <>
      {message && <p className={styles['skel-message']}>{message}</p>}
      {Array.from({ length: count }, (_, i) => (
        <SingleSkeleton key={i} />
      ))}
    </>
  );
}
