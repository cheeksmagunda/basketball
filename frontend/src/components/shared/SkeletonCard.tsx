import styles from './SkeletonCard.module.css';

interface SkeletonCardProps {
  count?: number;
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

export default function SkeletonCard({ count = 5 }: SkeletonCardProps) {
  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <SingleSkeleton key={i} />
      ))}
    </>
  );
}
